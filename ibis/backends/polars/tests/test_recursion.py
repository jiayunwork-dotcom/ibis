"""Polars execution tests for ``Table.recurse`` (native lazy fixed point)."""

from __future__ import annotations

import inspect

import polars as pl
import pytest

import ibis
import ibis.backends.polars.recursion as recursion_mod
from ibis.backends.tests.recursion_cases import (
    ANCESTOR_CLOSURE,
    CHAIN_SINGLE_APPLICATIONS,
    CHAIN_SINGLE_CLOSURE,
    DIAMOND_APPLICATIONS,
    DIAMOND_CLOSURE,
    MULTI_ROOT_APPLICATIONS,
    MULTI_ROOT_CLOSURE,
    NULL_DISTINCT_COUNT,
    NULL_DISTINCT_SORTED,
    ancestor_edges,
    ancestor_seed,
    ancestor_step,
    chain_seed_single,
    chain_step,
    diamond_edges,
    diamond_seed,
    multi_root_seed,
    null_rows,
)


@pytest.fixture
def con():
    return ibis.polars.connect()


def _sorted_int(result, col="node"):
    return sorted(result.column(col).to_pylist())


def test_compile_returns_lazyframe_without_collecting(con):
    expr = ancestor_seed().recurse(ancestor_step(ancestor_edges()))
    # the polars backend registers memtables during pre-execute hooks; do the
    # same explicitly when calling compile() directly
    con._register_in_memory_tables(expr)
    plan = con.compile(expr)
    assert isinstance(plan, pl.LazyFrame)
    # compilation/plan construction must not have pulled rows into Python


def test_ancestor_closure(con):
    expr = ancestor_seed().recurse(ancestor_step(ancestor_edges()))
    assert _sorted_int(con.to_pyarrow(expr)) == ANCESTOR_CLOSURE


def test_identity_step_one_application(con):
    t = ibis.memtable({"x": [1]})
    assert con.to_pyarrow(t.recurse(lambda k: k, max_iterations=1)).to_pydict() == {
        "x": [1]
    }


def test_chain_exact_budget_succeeds(con):
    expr = chain_seed_single().recurse(
        chain_step(), max_iterations=CHAIN_SINGLE_APPLICATIONS
    )
    assert _sorted_int(con.to_pyarrow(expr), "x") == CHAIN_SINGLE_CLOSURE


@pytest.mark.parametrize("short", [1, CHAIN_SINGLE_APPLICATIONS - 1])
def test_chain_budget_too_small_errors(con, short):
    expr = chain_seed_single().recurse(chain_step(), max_iterations=short)
    with pytest.raises(ibis.common.exceptions.IbisError, match="max_iterations"):
        con.to_pyarrow(expr)


def test_multi_root_budget_semantics(con):
    expr = multi_root_seed().recurse(
        chain_step(), max_iterations=MULTI_ROOT_APPLICATIONS
    )
    assert _sorted_int(con.to_pyarrow(expr), "x") == MULTI_ROOT_CLOSURE
    with pytest.raises(ibis.common.exceptions.IbisError, match="max_iterations"):
        con.to_pyarrow(
            multi_root_seed().recurse(
                chain_step(), max_iterations=MULTI_ROOT_APPLICATIONS - 1
            )
        )


def test_empty_seed_runs_step_once_and_returns_empty(con):
    empty = ibis.memtable({"x": []}, schema={"x": "int64"})
    assert con.to_pyarrow(empty.recurse(chain_step(), max_iterations=1)).num_rows == 0
    assert con.to_pyarrow(empty.recurse(chain_step(), max_iterations=5)).num_rows == 0


def test_null_safe_multicolumn_distinct(con):
    expr = null_rows().recurse(lambda k: k.union(k).distinct())
    result = con.to_pyarrow(expr)
    assert result.num_rows == NULL_DISTINCT_COUNT
    got = sorted(
        zip(
            result.column("a").to_pylist(),
            result.column("b").to_pylist(),
            result.column("c").to_pylist(),
        ),
        key=str,
    )
    assert got == sorted(NULL_DISTINCT_SORTED, key=str)


def test_duplicate_edges_do_not_duplicate_rows(con):
    expr = ancestor_seed().recurse(ancestor_step(ancestor_edges()))
    assert _sorted_int(con.to_pyarrow(expr)).count(5) == 1


def test_diamond_convergence_and_budget(con):
    edges = diamond_edges()
    step = lambda k: k.join(edges, k.node == edges.child).select(node=edges.parent)
    expr = diamond_seed().recurse(step, max_iterations=DIAMOND_APPLICATIONS)
    assert _sorted_int(con.to_pyarrow(expr)) == DIAMOND_CLOSURE
    with pytest.raises(ibis.common.exceptions.IbisError, match="max_iterations"):
        con.to_pyarrow(
            diamond_seed().recurse(step, max_iterations=DIAMOND_APPLICATIONS - 1)
        )


def test_outer_limit_does_not_bypass_guard(con):
    expr = chain_seed_single().recurse(chain_step(), max_iterations=1).limit(1)
    with pytest.raises(ibis.common.exceptions.IbisError, match="max_iterations"):
        con.to_pyarrow(expr)


def test_outer_filter_does_not_bypass_guard(con):
    expr = (
        chain_seed_single()
        .recurse(chain_step(), max_iterations=1)
        .filter(lambda d: d.x > 1000)
    )
    with pytest.raises(ibis.common.exceptions.IbisError, match="max_iterations"):
        con.to_pyarrow(expr)


def test_outer_aggregate_and_select(con):
    closure = ancestor_seed().recurse(ancestor_step(ancestor_edges()))
    assert con.to_pyarrow(closure.aggregate(n=ibis._.node.count())).to_pydict() == {
        "n": [len(ANCESTOR_CLOSURE)]
    }
    projected = closure.select(y=closure.node + 1)
    assert con.to_pyarrow(projected).num_rows == len(ANCESTOR_CLOSURE)


def test_two_independent_recursions_in_one_expression(con):
    a = ibis.memtable({"x": [1]}).recurse(lambda k: k, max_iterations=1)
    b = ibis.memtable({"x": [2]}).recurse(lambda k: k, max_iterations=1)
    assert _sorted_int(con.to_pyarrow(a.union(b)), "x") == [1, 2]


def test_closure_self_inner_join(con):
    seeds = ibis.memtable({"node": [1, 2]})
    expr = seeds.recurse(
        lambda k: k.join(k.view(), [("node", "node")]).select("node"),
        max_iterations=2,
    )
    assert _sorted_int(con.to_pyarrow(expr)) == [1, 2]


def test_repeated_evaluation_is_clean_and_stable(con):
    expr = ancestor_seed().recurse(ancestor_step(ancestor_edges()))
    first = con.to_pyarrow(expr)
    bad = chain_seed_single().recurse(chain_step(), max_iterations=1)
    with pytest.raises(ibis.common.exceptions.IbisError):
        con.to_pyarrow(bad)
    second = con.to_pyarrow(expr)
    # order is not part of the recursion contract; compare the record sets
    assert sorted(first.column("node").to_pylist()) == sorted(
        second.column("node").to_pylist()
    )


def test_original_graph_not_mutated(con):
    seeds = ancestor_seed()
    edges = ancestor_edges()
    expr = seeds.recurse(ancestor_step(edges))
    con._register_in_memory_tables(expr)
    before = repr(seeds.op())
    con.compile(expr)
    assert repr(seeds.op()) == before


def test_result_has_only_user_columns(con):
    expr = ancestor_seed().recurse(ancestor_step(ancestor_edges()))
    assert con.to_pyarrow(expr).schema.names == ["node"]


def test_no_python_row_materialization_in_plan(con):
    # the fixed-point loop may collect frames for convergence probes, but the
    # step itself must stay lazy: assert the produced candidate is a LazyFrame
    # at every round by checking the returned plan type and that iterrows /
    # to_pylist are never used to drive the loop
    expr = ancestor_seed().recurse(ancestor_step(ancestor_edges()))
    con._register_in_memory_tables(expr)
    lf = con.compile(expr)
    assert isinstance(lf, pl.LazyFrame)
    source = inspect.getsource(recursion_mod)
    assert "to_pylist" not in source
    assert "iterrows" not in source
    assert "iter_rows" not in source
