"""DuckDB execution tests for ``Table.recurse`` (WITH RECURSIVE compilation)."""

from __future__ import annotations

import re

import duckdb
import pytest

import ibis
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
    return ibis.duckdb.connect()


def _sorted_int(result, col="node"):
    return sorted(result.column(col).to_pylist())


def test_compile_produces_single_recursive_sql_statement(con):
    edges = ancestor_edges()
    expr = ancestor_seed().recurse(ancestor_step(edges))
    sql = con.compile(expr)
    assert isinstance(sql, str)
    assert sql.count("WITH RECURSIVE") == 1
    # compilation does not execute anything
    assert "error" in sql.lower() or "ERROR" in sql


def test_compile_only_does_not_require_running(con):
    # a compile-only call must succeed and remain a pure string
    expr = chain_seed_single().recurse(chain_step())
    sql = con.compile(expr)
    assert "WITH RECURSIVE" in sql


def test_ancestor_closure(con):
    edges = ancestor_edges()
    expr = ancestor_seed().recurse(ancestor_step(edges))
    assert _sorted_int(con.to_pyarrow(expr)) == ANCESTOR_CLOSURE


def test_identity_step_one_application(con):
    t = ibis.memtable({"x": [1]})
    result = con.to_pyarrow(t.recurse(lambda k: k, max_iterations=1))
    assert result.to_pydict() == {"x": [1]}


def test_chain_exact_budget_succeeds(con):
    expr = chain_seed_single().recurse(
        chain_step(), max_iterations=CHAIN_SINGLE_APPLICATIONS
    )
    assert _sorted_int(con.to_pyarrow(expr), "x") == CHAIN_SINGLE_CLOSURE


@pytest.mark.parametrize("short", [1, CHAIN_SINGLE_APPLICATIONS - 1])
def test_chain_budget_too_small_errors(con, short):
    expr = chain_seed_single().recurse(chain_step(), max_iterations=short)
    with pytest.raises(duckdb.InvalidInputException, match="max_iterations"):
        con.to_pyarrow(expr)


def test_multi_root_budget_semantics(con):
    expr = multi_root_seed().recurse(
        chain_step(), max_iterations=MULTI_ROOT_APPLICATIONS
    )
    assert _sorted_int(con.to_pyarrow(expr), "x") == MULTI_ROOT_CLOSURE
    with pytest.raises(duckdb.InvalidInputException, match="max_iterations"):
        con.to_pyarrow(
            multi_root_seed().recurse(
                chain_step(), max_iterations=MULTI_ROOT_APPLICATIONS - 1
            )
        )


def test_empty_seed_runs_step_once_and_returns_empty(con):
    empty = ibis.memtable({"x": []}, schema={"x": "int64"})
    expr = empty.recurse(chain_step(), max_iterations=1)
    assert con.to_pyarrow(expr).num_rows == 0
    # a larger budget on an empty seed still converges cleanly
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
    # the 4 -> 5 duplicate edge must still yield a single 5
    assert _sorted_int(con.to_pyarrow(expr)).count(5) == 1


def test_diamond_convergence_and_budget(con):
    edges = diamond_edges()
    step = lambda k: k.join(edges, k.node == edges.child).select(node=edges.parent)
    # exact budget succeeds
    expr = diamond_seed().recurse(step, max_iterations=DIAMOND_APPLICATIONS)
    assert _sorted_int(con.to_pyarrow(expr)) == DIAMOND_CLOSURE
    # one fewer fails because the closure is still growing
    with pytest.raises(duckdb.InvalidInputException, match="max_iterations"):
        con.to_pyarrow(
            diamond_seed().recurse(step, max_iterations=DIAMOND_APPLICATIONS - 1)
        )


def test_outer_limit_does_not_bypass_guard(con):
    expr = chain_seed_single().recurse(chain_step(), max_iterations=1).limit(1)
    with pytest.raises(duckdb.InvalidInputException, match="max_iterations"):
        con.to_pyarrow(expr)


def test_outer_filter_does_not_bypass_guard(con):
    expr = (
        chain_seed_single()
        .recurse(chain_step(), max_iterations=1)
        .filter(lambda d: d.x > 1000)
    )
    with pytest.raises(duckdb.InvalidInputException, match="max_iterations"):
        con.to_pyarrow(expr)


def test_outer_column_projection_does_not_bypass_guard(con):
    expr = chain_seed_single().recurse(chain_step(), max_iterations=1).select("x")
    with pytest.raises(duckdb.InvalidInputException, match="max_iterations"):
        con.to_pyarrow(expr)


def test_outer_aggregate_and_select(con):
    edges = ancestor_edges()
    closure = ancestor_seed().recurse(ancestor_step(edges))
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
    # after a failed evaluation, a valid one still succeeds
    bad = chain_seed_single().recurse(chain_step(), max_iterations=1)
    with pytest.raises(duckdb.InvalidInputException):
        con.to_pyarrow(bad)
    second = con.to_pyarrow(expr)
    assert first == second


def test_original_graph_not_mutated(con):
    seeds = ancestor_seed()
    edges = ancestor_edges()
    before = repr(seeds.op())
    expr = seeds.recurse(ancestor_step(edges))
    con.compile(expr)
    assert repr(seeds.op()) == before


def test_result_has_only_user_columns(con):
    expr = ancestor_seed().recurse(ancestor_step(ancestor_edges()))
    result = con.to_pyarrow(expr)
    assert result.schema.names == ["node"]


def test_plan_not_expanded_to_max_iterations(con):
    # the generated SQL contains a fixed number of recursive CTE definitions
    # regardless of the budget: the graph is never unrolled one copy per
    # budgeted iteration
    def normalized(budget):
        # budgets above 5 cannot collide with the step's own "<= 5" literal,
        # so a plain integer replacement normalizes exactly the budget
        sql = con.compile(
            chain_seed_single().recurse(chain_step(), max_iterations=budget)
        )
        sql = re.sub(r"ibis_pandas_memtable_[a-z0-9]+", "MEM", sql)
        sql = re.sub(r"ibis_recurse_\d+", "ibis_recurse_N", sql)
        return sql.replace(str(budget), "BUDGET")

    assert normalized(7) == normalized(99)
    large = con.compile(chain_seed_single().recurse(chain_step(), max_iterations=500))
    # exactly one closure and one depth CTE *definition*, not one per
    # iteration (their names recur in the step and guard, so count AS clauses)
    assert large.count('_closure" AS') == 1
    assert large.count('_depth" AS') == 1
