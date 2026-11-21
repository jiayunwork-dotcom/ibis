"""Native Polars evaluation of recursive relations.

Unlike the SQL backend, which emits a ``WITH RECURSIVE`` CTE and lets the
database iterate, Polars has no recursive relation. The fixed point is
therefore driven here in Python, but every round stays a *lazy* relation plan:
candidate rows are never materialized into Python lists. The only collected
data are the small anti-join probes needed to test convergence and the final
closure; even those move as Arrow/polars frames, not row sequences.

The placeholder always receives the full accumulated closure (a
``LazyFrame``), so a step depending on the whole discovered set -- not merely
the last increment -- behaves identically to the SQL backend.
"""

from __future__ import annotations

import polars as pl

import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.sql.rewrites import (
    RecursivePlaceholderRef,
    lower_recursion,
)

# Active marker -> LazyFrame bindings for the step currently being translated.
# The polars singledispatch rules call the global ``translate`` recursively, so
# the marker translation rule consults this module-level mapping rather than
# carrying the substitution through every call signature. The binding is reset
# in a finally block, so a failed evaluation cannot leak state into the next.
_PLACEHOLDER_FRAMES: dict = {}
# uid -> original RecursiveRelation, needed when a step nests another recursion
_RECURSION_BY_UID: dict = {}


def current_placeholder_frames():
    return _PLACEHOLDER_FRAMES


def set_placeholder_frames(mapping):
    # module-level state is intentionally rebound here; callers are confined
    # to evaluate_recursion, which always restores the previous binding
    globals()["_PLACEHOLDER_FRAMES"] = mapping


def evaluate_recursion(
    op: ops.RecursiveRelation,
    *,
    ctx: pl.SQLContext,
    translate,
):
    """Run the fixed-point loop for ``op`` and return a ``LazyFrame``.

    The returned frame is the distinct closure over the user columns. Budget
    exhaustion raises :class:`IbisError` rather than returning a partial set.
    """
    budget = op.max_iterations
    cols = op.schema.names

    seed_lf = translate(op.seed, ctx=ctx)
    # normalize to the user-column layout and dedup the seed
    closure = seed_lf.select(cols).unique(keep="any")

    # pre-lower the step once: the same marker graph is rebound each round to
    # the then-current closure, but the structural rewrite is reused.
    def name_for(uid):
        return f"ibis_recurse_{uid}"

    lowered_step, _ = lower_recursion(op.step, name_for=name_for)
    _RECURSION_BY_UID[op.placeholder.uid] = op
    for inner in op.find(ops.RecursiveRelation):
        _RECURSION_BY_UID[inner.placeholder.uid] = inner

    applications = 0
    previous_binding = dict(current_placeholder_frames())
    try:
        while applications < budget:
            applications += 1
            known = closure

            # one LazyFrame handle per marker usage so the two sides of a
            # self join stay distinct plans even though they read one set
            markers = lowered_step.find(RecursivePlaceholderRef)
            binding = {
                marker: known
                if i == 0
                else known.select(known.collect_schema().names())
                for i, marker in enumerate(markers)
            }
            set_placeholder_frames(binding)

            candidates = translate(lowered_step, ctx=ctx).select(cols)

            # NULL-safe anti join on every user column (join_nulls=True makes
            # NULL == NULL, matching SQL IS NOT DISTINCT FROM)
            new_rows = candidates.join(known, on=cols, how="anti", join_nulls=True)

            if new_rows.collect().is_empty():
                # final application added nothing: successful convergence
                return closure

            # The closure is the finite *distinct* record set (NULLs compare
            # equal across the whole row under polars' default unique()).
            # Materializing it back into a fresh LazyFrame keeps the plan
            # shallow: without this, each round would nest another concat and
            # re-translate an ever deeper graph, turning a long chain into a
            # super-linear rebuild. Only the small current closure is
            # materialized; the step itself stays a lazy plan every round.
            closure = (
                pl.concat([closure, new_rows], how="vertical_relaxed")
                .unique(keep="any")
                .collect()
                .lazy()
            )
    finally:
        set_placeholder_frames(previous_binding)

    # budget spent and the last application still produced new rows
    raise com.IbisError(f"recursion did not converge within max_iterations={budget}")
