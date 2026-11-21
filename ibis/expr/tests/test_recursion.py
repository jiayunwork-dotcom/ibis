"""Construction-time validation for ``Table.recurse``.

These tests never touch a backend: they check that the recursive relation is
built correctly, that misconfiguration and unsupported step shapes raise
``IbisInputError`` while the graph is being constructed, and that recursive
references are scoped by identity rather than table names.
"""

from __future__ import annotations

import pytest

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis.expr.recursion import _count_placeholder_occurrences


@pytest.fixture
def seeds():
    return ibis.table({"node": "int64"}, name="seeds")


@pytest.fixture
def edges():
    return ibis.table({"child": "int64", "parent": "int64"}, name="edges")


def test_recurse_builds_recursive_relation(seeds, edges):
    expr = seeds.recurse(
        lambda known: known.join(edges, known.node == edges.child).select(
            node=edges.parent
        )
    )
    op = expr.op()
    assert isinstance(op, ops.RecursiveRelation)
    assert isinstance(op.placeholder, ops.RecursivePlaceholder)
    assert op.max_iterations == 1000
    assert expr.schema() == seeds.schema()
    # the placeholder is referenced exactly once in the step
    assert _count_placeholder_occurrences(op.step, op.placeholder.uid) == 1


def test_step_is_called_once_and_not_executed(seeds):
    calls = []

    def step(known):
        calls.append(known)
        return known

    seeds.recurse(step)
    assert len(calls) == 1
    assert isinstance(calls[0].op(), ops.RecursivePlaceholder)


def test_identity_step_is_legal(seeds):
    expr = seeds.recurse(lambda known: known)
    assert isinstance(expr.op(), ops.RecursiveRelation)


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_max_iterations_must_be_positive(seeds, bad):
    with pytest.raises(com.IbisInputError, match="positive integer"):
        seeds.recurse(lambda k: k, max_iterations=bad)


@pytest.mark.parametrize("bad", [True, False, 1.0, 2.5, "3", None])
def test_max_iterations_rejects_non_int_including_bool(seeds, bad):
    with pytest.raises(com.IbisInputError, match="positive integer"):
        seeds.recurse(lambda k: k, max_iterations=bad)


def test_step_must_reference_placeholder(seeds, edges):
    with pytest.raises(com.IbisInputError, match="at least once"):
        seeds.recurse(lambda k: edges.select(node=edges.parent))


def test_missing_placeholder_in_projection(seeds):
    with pytest.raises(com.IbisInputError, match="at least once"):
        seeds.recurse(lambda k: ibis.table({"node": "int64"}, name="other"))


def test_nested_recursion_rejected(seeds):
    inner = seeds.recurse(lambda k: k)
    with pytest.raises(com.IbisInputError, match="nested recursion"):
        seeds.recurse(lambda k: k.union(inner))


def test_foreign_placeholder_rejected(seeds):
    other = ibis.table({"node": "int64"}, name="other")
    foreign = other.recurse(lambda k: k).op().placeholder
    with pytest.raises(com.IbisInputError, match="another recursion"):
        seeds.recurse(lambda k: k.union(foreign.to_expr()))


def test_same_named_tables_get_distinct_placeholders():
    a = ibis.table({"node": "int64"}, name="same")
    b = ibis.table({"node": "int64"}, name="same")
    assert (
        a.recurse(lambda k: k).op().placeholder.uid
        != b.recurse(lambda k: k).op().placeholder.uid
    )


def test_limit_rejected(seeds):
    with pytest.raises(com.IbisInputError, match="limit"):
        seeds.recurse(lambda k: k.limit(1))


def test_order_by_rejected(seeds):
    with pytest.raises(com.IbisInputError, match="ordering"):
        seeds.recurse(lambda k: k.order_by("node"))


def test_aggregate_rejected(seeds):
    with pytest.raises(com.IbisInputError, match="aggregation"):
        seeds.recurse(lambda k: k.aggregate(ibis._.node.count()))


def test_window_rejected(seeds):
    with pytest.raises(com.IbisInputError, match="window"):
        seeds.recurse(lambda k: k.mutate(rn=ibis.row_number()))


def test_reduction_rejected(seeds):
    with pytest.raises(com.IbisInputError, match=r"window|reduction"):
        seeds.recurse(lambda k: k.mutate(c=ibis._.node.sum()))


def test_random_expression_rejected(seeds):
    with pytest.raises(com.IbisInputError, match="non-deterministic"):
        seeds.recurse(lambda k: k.mutate(r=ibis.random()))


def test_unsupported_column_type_rejected_on_seed():
    t = ibis.table({"node": "float64"}, name="t")
    with pytest.raises(com.IbisInputError, match="int64, string and boolean"):
        t.recurse(lambda k: k)


def test_step_column_type_mismatch_rejected():
    t = ibis.table({"node": "string"}, name="t")
    with pytest.raises(com.IbisInputError, match="base type"):
        t.recurse(lambda k: k.mutate(node=k.node.cast("int64")))


def test_step_column_order_mismatch_rejected():
    t = ibis.table({"a": "int64", "b": "int64"}, name="t")
    with pytest.raises(com.IbisInputError, match="name and order"):
        t.recurse(lambda k: k.select("b", "a"))


def test_left_join_of_two_placeholder_views_rejected(seeds):
    with pytest.raises(com.IbisInputError, match="inner"):
        seeds.recurse(
            lambda k: k.left_join(k.view(), [("node", "node")]).select("node")
        )


def test_inner_self_join_is_legal(seeds):
    expr = seeds.recurse(lambda k: k.join(k.view(), [("node", "node")]).select("node"))
    op = expr.op()
    assert _count_placeholder_occurrences(op.step, op.placeholder.uid) == 2


def test_cross_self_join_is_legal(seeds):
    expr = seeds.recurse(lambda k: k.cross_join(k.view()).select("node").distinct())
    op = expr.op()
    assert _count_placeholder_occurrences(op.step, op.placeholder.uid) == 2


def test_left_join_with_fixed_relation_is_legal(seeds, edges):
    expr = seeds.recurse(
        lambda k: k.left_join(edges, k.node == edges.child).select(node=k.node)
    )
    assert isinstance(expr.op(), ops.RecursiveRelation)


def test_more_than_two_placeholder_uses_rejected(seeds):
    def step(k):
        joined = k.join(k.view(), [("node", "node")]).select("node")
        return joined.union(k).select("node")

    with pytest.raises(com.IbisInputError, match="at most twice"):
        seeds.recurse(step)


def test_distinct_and_union_are_legal(seeds):
    seeds.recurse(lambda k: k.union(k).distinct())
    seeds.recurse(lambda k: k.distinct())


def test_nullable_merged_schema():
    t = ibis.table({"a": "int64", "b": "boolean"}, name="t")
    expr = t.recurse(lambda k: k.mutate(a=ibis.literal(None, "int64")))
    schema = expr.schema()
    # the null-producing column stays nullable; unbound-table columns are
    # nullable by default anyway
    assert schema["a"].nullable is True
    assert schema["b"].nullable is True

    # with an explicitly non-nullable seed column, nullability is preserved
    non_null = ibis.table({"a": dt.Int64(nullable=False)}, name="nn")
    out = non_null.recurse(lambda k: k)
    assert out.schema()["a"].nullable is False


def test_seed_graph_is_not_mutated(seeds, edges):
    before = repr(seeds.op())
    seeds.recurse(
        lambda known: known.join(edges, known.node == edges.child).select(
            node=edges.parent
        )
    )
    assert repr(seeds.op()) == before


def test_supported_types_all_accepted():
    t = ibis.table({"a": "int64", "b": "string", "c": "boolean"}, name="t")
    expr = t.recurse(lambda k: k.mutate(c=~k.c).select("a", "b", "c"))
    assert set(expr.schema().names) == {"a", "b", "c"}


def test_outer_operations_construct_without_loop(seeds, edges):
    r = seeds.recurse(
        lambda known: known.join(edges, known.node == edges.child).select(
            node=edges.parent
        )
    )
    # these used to loop in DerefMap.backtrack with self-referential fields
    r.filter(lambda d: d.node > 0)
    r.select("node")
    r.aggregate(ibis._.node.count())
    r.alias("a")
    r.order_by("node").limit(3)
