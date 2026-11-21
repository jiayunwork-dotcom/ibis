"""Compilation of recursive relations into SQL ``WITH RECURSIVE`` CTEs.

The module only builds a sqlglot expression graph -- it never executes SQL and
never reads data. The recursion itself runs inside the database engine.

Why two cooperative CTEs
------------------------
DuckDB's recursive CTE evaluates some shapes (anti joins, ``NOT IN``,
``EXCEPT``) against a second reference of the recursive name using only the
last working table rather than the accumulated closure, so a single CTE cannot
both apply the step to the full discovered set and deduplicate across rounds.
The builder therefore emits two CTEs that share one compiled step body:

* ``..._closure`` -- a ``UNION`` (distinct) CTE on the user columns only. The
  engine's own distinct-set semantics (NULLs in the same position compare
  equal) give the finite record set and make cycles terminate naturally. It
  applies the step to the whole closure, never just the last increment.
* ``..._depth`` -- a ``UNION ALL`` CTE that replays the same step one round at
  a time against the current frontier and records the round number. It is
  bounded by the closure's cardinality, which is a hard upper bound on the
  number of discovery rounds, so it cannot run away on a cyclic graph.

The minimum depth at which each distinct record first appears determines how
many step applications were productive; adding one for the mandatory final
application that adds nothing gives the exact application count. A guard
cross-joined to the result raises ``error()`` when that count exceeds
``max_iterations``. Because the guard is an ordinary predicate of the final
read, outer ``LIMIT``/filter pushdown cannot bypass it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlglot as sg
import sqlglot.expressions as sge

import ibis.expr.operations as ops

if TYPE_CHECKING:
    from ibis.backends.sql.compilers.base import SQLGlotCompiler

DEPTH_COL = "__ibis_recurse_depth__"


def _column(name, table=None, quoted=True):
    return sg.column(name, table=table, quoted=quoted)


def _boolean(value):
    return sge.Boolean(this=value)


class RecursiveCTEBuilder:
    """Turns :class:`RecursiveRelation` nodes into recursive CTE sqlglot ASTs."""

    def __init__(self, compiler: SQLGlotCompiler):
        self.compiler = compiler
        # uid -> (closure cte name, depth cte name)
        self.names: dict[int, tuple[str, str]] = {}
        # uid -> compiled closure/depth definitions
        self.closure_defs: dict[int, sge.Expression] = {}
        self.depth_defs: dict[int, sge.Expression] = {}
        # uid -> budget
        self.budgets: dict[int, int] = {}
        # uid -> user columns in declaration order
        self.columns: dict[int, list[str]] = {}

    def names_for(self, uid: int) -> tuple[str, str]:
        if uid not in self.names:
            self.names[uid] = (
                f"ibis_recurse_{uid}_closure",
                f"ibis_recurse_{uid}_depth",
            )
        return self.names[uid]

    # kept for callers that only need a single name (CTE attachment)
    def name_for(self, uid: int) -> str:
        return self.names_for(uid)[0]

    def build(self, op: ops.RecursiveRelation) -> None:
        """Compile ``op`` and every inner recursion it contains."""
        uid = op.placeholder.uid
        if uid in self.closure_defs:
            return

        for inner in op.find(ops.RecursiveRelation):
            if inner is not op:
                self.build(inner)

        closure_name, depth_name = self.names_for(uid)
        cols = list(op.schema.names)
        budget = op.max_iterations
        quoted = self.compiler.quoted
        self.columns[uid] = cols
        self.budgets[uid] = budget

        seed_sql = self.compiler.translate_relation_subgraph(op.seed)
        seed_select = self._as_select(seed_sql)

        # ---- closure CTE: anchor UNION (distinct) step(closure) ----------
        closure_view = sg.table(closure_name, quoted=quoted)
        closure_step = self.compiler.translate_recursive_step(
            op.step,
            placeholder=op.placeholder,
            placeholder_sql=sg.select(sg.exp.Star(), copy=False).from_(
                closure_view, copy=False
            ),
            cte_name=closure_name,
        )
        closure_body = sg.union(
            seed_select,
            self._as_select(closure_step),
            distinct=True,
            copy=False,
        )

        # ---- depth CTE: replay the step per frontier round ---------------
        # The recursive name in the recursive term resolves to the *working
        # table* (the previous round's rows), which is exactly the frontier
        # semantics needed: each round expands only the rows added last round.
        # The round number is carried directly as parent_round + 1 instead of
        # a scalar max() rescan, keeping a long chain linear. The CTE is
        # bounded by the literal budget (never the closure size) so no extra
        # full-table subquery is evaluated per round.
        depth_view = sg.table(depth_name, quoted=quoted)
        # The depth CTE's placeholder exposes *user columns only*: the step
        # never sees the round number, so self joins cannot duplicate it. The
        # next round number is computed outside the step from the recursive
        # working table, whose rows all share the previous round's number.
        depth_placeholder = sg.select(*[_column(c) for c in cols], copy=False).from_(
            depth_view.copy(), copy=False
        )
        depth_step = self.compiler.translate_recursive_step(
            op.step,
            placeholder=op.placeholder,
            placeholder_sql=depth_placeholder,
            cte_name=depth_name,
        )
        depth_step = self._as_select(depth_step)
        # dedup candidate rows within the term so a cyclic graph cannot make
        # the depth table grow with the number of paths to the same record
        depth_step.set("distinct", sge.Distinct())

        # next round = one past the current working table's round. A single
        # non-join reference is cheap and stays linear; join-shaped steps keep
        # their own recursive references inside ``depth_step``.
        prev_round = sge.Subquery(
            this=sg.select(sge.Max(this=_column(DEPTH_COL)), copy=False).from_(
                depth_view.copy(), copy=False
            )
        )
        depth_recursive = (
            sg.select(
                sge.Add(this=prev_round, expression=sge.convert(1)).as_(
                    DEPTH_COL, quoted=quoted
                ),
                *[_column(c).as_(c, quoted=quoted) for c in cols],
                copy=False,
            )
            .from_(depth_step.subquery("step_round", copy=False), copy=False)
            .where(
                sge.LT(this=prev_round.copy(), expression=sge.convert(budget)),
                copy=False,
            )
        )

        depth_anchor = sg.select(
            sge.convert(0).as_(DEPTH_COL, quoted=quoted),
            *seed_select.expressions,
            copy=False,
        )
        for key in ("from", "where", "joins", "group", "qualify", "distinct"):
            depth_anchor.set(key, seed_select.args.get(key))

        depth_body = sg.union(depth_anchor, depth_recursive, distinct=False, copy=False)

        self.closure_defs[uid] = closure_body
        self.depth_defs[uid] = depth_body

    # -- convergence guard ------------------------------------------------
    def guard_predicate(self, uid: int):
        """Raise when more than ``max_iterations`` step applications were needed.

        Returned as a scalar boolean subquery so it is self-contained: it can
        be dropped into any WHERE clause without depending on the enclosing
        query's FROM. Outer LIMIT/filter pushdown therefore cannot bypass it.
        """
        _closure_name, depth_name = self.names_for(uid)
        budget = self.budgets[uid]
        quoted = self.compiler.quoted
        cols = self.columns[uid]

        # productive applications = max first-seen depth; total applications
        # adds one for the final application that adds no records. On an empty
        # closure there are no first-seen rows: COALESCE keeps the mandatory
        # single application.
        first_seen_select = (
            sg.select(
                sge.Min(this=_column(DEPTH_COL)).as_("fd", quoted=quoted),
                copy=False,
            )
            .from_(sg.table(depth_name, quoted=quoted), copy=False)
            .group_by(*[_column(c) for c in cols], copy=False)
        )
        applications_select = sg.select(
            sge.Add(
                this=sge.Coalesce(
                    this=sge.Max(this=sg.column("fd", quoted=quoted)),
                    expression=sge.convert(0),
                ),
                expression=sge.convert(1),
            ).as_("applications", quoted=quoted),
            copy=False,
        ).from_(first_seen_select.subquery("first_seen", copy=False), copy=False)

        applications = sge.Subquery(this=applications_select)
        error_call = sge.Anonymous(
            this="error",
            expressions=[
                sge.Literal.string(
                    f"recursion did not converge within max_iterations={budget}"
                )
            ],
        )
        return sge.Case(
            ifs=[
                sge.If(
                    this=sge.GT(
                        this=applications.copy(),
                        expression=sge.convert(budget),
                    ),
                    true=error_call,
                )
            ],
            default=sg.exp.true(),
        )

    def closure_name(self, uid: int) -> str:
        return self.names_for(uid)[0]

    @staticmethod
    def _as_select(node):
        if isinstance(node, sge.Subquery):
            node = node.this
        if isinstance(node, sge.Table):
            return sg.select(sg.exp.Star(), copy=False).from_(node, copy=False)
        return node
