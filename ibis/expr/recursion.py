"""Construction and static validation of recursive relations.

A recursion is described by a *seed* relation (the starting set of records)
and a *step* relation built around a :class:`RecursivePlaceholder`. The
placeholder stands for the full set of distinct records discovered so far and
the step is applied to that whole set -- never just to the previous round's
increment -- until an application adds no new records.

The module deliberately performs every structural check while the expression
is being built (see :func:`build_recursion`), so that misconfiguration and
unsupported ``step`` shapes raise :class:`IbisInputError` at construction time
rather than failing deep inside a backend compiler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.expr.schema import Schema

if TYPE_CHECKING:
    import ibis.expr.types as ir

# Column types usable inside a recursive relation for now. No implicit casts
# are performed, so the seed and step must use exactly these base types.
_SUPPORTED_TYPES = (dt.Int64, dt.String, dt.Boolean)

# Deterministic, row-wise value operations. Everything derived from these is
# allowed as long as none of the forbidden node classes below shows up.
_ALLOWED_RELATIONS = (
    ops.Project,
    ops.Filter,
    ops.Distinct,
    ops.Set,
    ops.JoinChain,
    ops.Reference,
    ops.RecursivePlaceholder,
    ops.View,
    ops.PhysicalTable,
    ops.DummyTable,
    ops.FillNull,
    ops.DropNull,
    ops.TableUnnest,
)


def _validate_max_iterations(max_iterations: int) -> int:
    # bool is a subclass of int; reject it explicitly to avoid silently
    # interpreting True/False as 1/0.
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise com.IbisInputError(
            "max_iterations must be a positive integer, got "
            f"{type(max_iterations).__name__}"
        )
    if max_iterations <= 0:
        raise com.IbisInputError(
            f"max_iterations must be a positive integer, got {max_iterations}"
        )
    return max_iterations


def _validate_schema(schema, where: str) -> None:
    for name, dtype in schema.items():
        if not isinstance(dtype, _SUPPORTED_TYPES):
            raise com.IbisInputError(
                f"recursive relations only support int64, string and boolean "
                f"columns; column {name!r} in {where} has type {dtype}"
            )


def _check_schemas_match(seed_schema, step_schema) -> None:
    if step_schema.names != seed_schema.names:
        raise com.IbisInputError(
            "step result columns must match the seed table's columns in name "
            f"and order, got {step_schema.names} vs {seed_schema.names}"
        )
    for name in seed_schema.names:
        seed_type = seed_schema[name]
        step_type = step_schema[name]
        if type(seed_type) is not type(step_type):
            raise com.IbisInputError(
                f"step result column {name!r} has base type {step_type} but "
                f"the seed column has {seed_type}; no implicit conversion is "
                "performed in recursive relations"
            )


def _nullable_merge(seed_schema, step_schema) -> dt.Schema:
    # The closure must be able to hold NULLs produced by either side; a NULL
    # in the same position on both sides is treated as equal at dedup time.
    fields = {
        name: seed_schema[name].copy(
            nullable=seed_schema[name].nullable or step_schema[name].nullable
        )
        for name in seed_schema.names
    }
    return Schema(fields)


def _iter_relations(node):
    """Yield each relation node in ``node`` exactly once (structural sharing)."""
    seen = set()
    for rel in node.find(ops.Relation):
        if rel not in seen:
            seen.add(rel)
            yield rel


def _count_placeholder_occurrences(node, uid):
    """Count usage positions of the placeholder with ``uid``.

    Plain :meth:`Node.find` deduplicates structurally equal nodes, but the
    recursion limit is about *usage positions*: reusing the same placeholder
    on both sides of a self join counts as two references. The walk follows
    relation edges, additionally crossing non-relation container nodes (e.g.
    :class:`JoinLink`) that merely group relation arguments; value nodes --
    whose :class:`Field` children point back at the placeholder's columns --
    are never descended into.
    """
    count = 0
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, ops.RecursivePlaceholder):
            if current.uid == uid:
                count += 1
            # keep walking: the same placeholder node may be reached again
            # through another relation edge (the second side of a self join)
        for child in current.__children__:
            if isinstance(child, ops.Relation) or not isinstance(child, ops.Value):
                stack.append(child)
    return count


def _validate_step(step: ops.Relation, placeholder: ops.RecursivePlaceholder) -> None:
    # 1. nested recursion is rejected outright, before looking at placeholder
    # ownership (an inner recursion carries its own, differently scoped
    # placeholder).
    if step.find(ops.RecursiveRelation):
        raise com.IbisInputError(
            "nested recursion is not allowed inside a recurse step"
        )

    # 2. placeholder ownership: only this recursion's placeholder may appear.
    placeholders = [
        rel
        for rel in _iter_relations(step)
        if isinstance(rel, ops.RecursivePlaceholder)
    ]
    foreign = [p for p in placeholders if p.uid != placeholder.uid]
    if foreign:
        raise com.IbisInputError(
            "step references a recursive placeholder belonging to another "
            "recursion; recursive references are scoped to the Table.recurse "
            "call that created them"
        )
    if not placeholders:
        raise com.IbisInputError(
            "step must reference the recursive placeholder table at least "
            "once; a step without a recursive reference cannot grow the "
            "closure"
        )

    # Count actual usage positions rather than distinct nodes: a self-join on
    # the same placeholder node uses two join sides and therefore counts as
    # two references.
    occurrences = _count_placeholder_occurrences(step, placeholder.uid)
    if occurrences > 2:
        raise com.IbisInputError(
            "step may reference the recursive placeholder at most twice; "
            f"found {occurrences} usage positions"
        )

    # 3. relation-level allow list and explicit feature bans.
    for rel in _iter_relations(step):
        if isinstance(rel, ops.RecursivePlaceholder):
            continue
        if isinstance(rel, ops.Aggregate):
            raise com.IbisInputError(
                "grouping/aggregation is not supported inside a recurse step"
            )
        if isinstance(rel, ops.Sort):
            raise com.IbisInputError(
                "ordering (sort) is not supported inside a recurse step"
            )
        if isinstance(rel, ops.Limit):
            raise com.IbisInputError("limit is not supported inside a recurse step")
        if isinstance(rel, ops.Sample):
            raise com.IbisInputError(
                "sampling (including random samples) is not supported inside "
                "a recurse step"
            )
        if not isinstance(rel, _ALLOWED_RELATIONS):
            raise com.IbisInputError(
                f"relation operation {type(rel).__name__} is not supported "
                "inside a recurse step"
            )

    # Window functions (including reductions turned into windows) and bare
    # reductions are rejected, as are non-deterministic expressions. Note that
    # traversal must not be filtered to values: value nodes live below
    # relation nodes and a value filter would prune the whole graph.
    if step.find(ops.WindowFunction):
        raise com.IbisInputError(
            "window functions are not supported inside a recurse step"
        )
    if step.find(ops.Reduction):
        raise com.IbisInputError(
            "aggregating reductions are not supported inside a recurse step"
        )
    if step.find(ops.Impure):
        raise com.IbisInputError(
            "non-deterministic (e.g. random) expressions are not supported "
            "inside a recurse step"
        )

    # 3. join shape: left joins between recursive placeholder views (and any
    # join producing NULL rows from a recursive side other than an inner
    # self-join) are banned.
    _validate_joins(step, placeholder)


def _placeholder_refs(rel: ops.Relation, uid: int) -> bool:
    """Whether relation ``rel`` contains a placeholder with ``uid``."""
    return any(
        isinstance(node, ops.RecursivePlaceholder) and node.uid == uid
        for node in rel.find(ops.RecursivePlaceholder)
    )


def _validate_joins(step: ops.Relation, placeholder: ops.RecursivePlaceholder) -> None:
    uid = placeholder.uid
    for join in step.find(ops.JoinChain):
        sides = [join.first, *(link.table for link in join.rest)]
        recursive_sides = [side for side in sides if _placeholder_refs(side, uid)]
        if not recursive_sides:
            # join of fixed relations only: fine
            continue

        for link in join.rest:
            if link.how in {"inner", "cross"}:
                # cross joins do not introduce NULL rows on either side, so
                # they are as safe as inner joins for self-combination
                continue
            # any non-inner/non-cross join whose *added* side contains the
            # recursive placeholder is banned (left join between placeholder
            # views, right/outer joins pulling placeholder rows, ...). A left
            # join with the placeholder on the preserved side and a purely
            # fixed right side is a regular fixed-relation lookup.
            if _placeholder_refs(link.table, uid):
                raise com.IbisInputError(
                    "only inner joins are permitted when adding a recursive "
                    "placeholder view to a join within a recurse step, got "
                    f"{link.how!r} join"
                )

        if len(recursive_sides) > 2:
            raise com.IbisInputError(
                "a recurse step join may use the recursive placeholder at most twice"
            )
        if len(recursive_sides) == 2:
            # self-combination of discovered relations: only inner/cross is
            # allowed (no NULL-producing join kinds)
            if any(link.how not in ("inner", "cross") for link in join.rest):
                raise com.IbisInputError(
                    "self joins of the recursive placeholder must be inner joins"
                )


def build_recursion(
    seed: ir.Table,
    step,
    max_iterations: int,
) -> ops.RecursiveRelation:
    """Validate inputs and construct a :class:`RecursiveRelation`.

    ``step`` is invoked exactly once, with a placeholder table having the
    same schema as the seed. No data is read: the callback only builds the
    relation graph.
    """
    max_iterations = _validate_max_iterations(max_iterations)

    if not isinstance(seed.op(), ops.Relation):
        raise com.IbisInputError("recurse must be called on a table expression")

    seed_schema = seed.schema()
    _validate_schema(seed_schema, "the seed table")

    placeholder_op = ops.RecursivePlaceholder.create(schema=seed_schema)
    placeholder_expr = placeholder_op.to_expr()

    try:
        step_expr = step(placeholder_expr)
    except com.IbisError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise com.IbisInputError(
            f"step callable raised {type(exc).__name__}: {exc}"
        ) from exc

    if not isinstance(step_expr, type(seed)):
        raise com.IbisInputError(
            "step must return a table expression with the same structure as "
            f"the seed table, got {type(step_expr).__name__}"
        )

    step_op = step_expr.op()
    step_schema = step_op.schema

    # structural validation of the step (placeholder usage, banned features)
    # happens before schema/type checks so that misconfigured steps get the
    # dedicated error message
    _validate_step(step_op, placeholder_op)

    _validate_schema(step_schema, "the step result")
    _check_schemas_match(seed_schema, step_schema)

    # The closure must be able to hold NULLs produced by either side. Rebuild
    # the placeholder (and the step graph pointing at it) with the loosened
    # schema; the uid stays the same so placeholder ownership is preserved.
    merged_schema = _nullable_merge(seed_schema, step_schema)
    if merged_schema != placeholder_op.schema:
        new_placeholder = ops.RecursivePlaceholder(
            schema=merged_schema, uid=placeholder_op.uid
        )
        step_op = step_op.replace(
            {
                old: new_placeholder
                for old in step_op.find(ops.RecursivePlaceholder)
                if old.uid == placeholder_op.uid
            }
        )
        placeholder_op = new_placeholder

    return ops.RecursiveRelation(
        seed=seed.op(),
        step=step_op,
        placeholder=placeholder_op,
        max_iterations=max_iterations,
    )
