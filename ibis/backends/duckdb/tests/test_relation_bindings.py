"""Tests for per-call relation bindings (``table_bindings``).

Relation bindings let one expression built from ``ibis.table()`` placeholders
be executed repeatedly against different ``pyarrow.Table`` batches without
creating or dropping catalog tables. The tests below cover:

* alternating batches on the same expression and shadowing of same-named
  catalog tables;
* the four entry points on both the backend and the expression;
* validation of keys, values, columns, types and nullability;
* structural node identity (same-name/different-schema placeholders,
  shared subgraphs and view self-joins);
* every supported query shape (CTE reuse, aggregation, window, join,
  correlated exists);
* scalar ``params`` combined with bindings, result shapes and limits;
* deterministic resource cleanup (eager, streaming, error and reuse paths);
* compile/execute identity without catalog side effects.
"""

from __future__ import annotations

import gc

import duckdb
import pytest

import ibis
import ibis.expr.operations as ops
from ibis import udf
from ibis.backends.duckdb.bindings import BindingSession
from ibis.common.exceptions import IbisError, IbisInputError

pa = pytest.importorskip("pyarrow")


@pytest.fixture
def con():
    return ibis.duckdb.connect()


# ---------------------------------------------------------------------------
# basic execution
# ---------------------------------------------------------------------------


def test_alternating_batches_same_expression(con):
    t = ibis.table(dict(a="int64", b="string"), name="t")
    expr = t.filter(t.a > 1).select("a", "b").order_by("a")

    batch1 = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    batch2 = pa.table({"a": [10, 20], "b": ["p", "q"]})

    r1 = con.execute(expr, table_bindings={t: batch1})
    r2 = con.execute(expr, table_bindings={t: batch2})

    assert r1.a.tolist() == [2, 3]
    assert r2.a.tolist() == [10, 20]
    # catalog was never touched
    assert con.list_tables() == []


def test_same_named_catalog_table_is_shadowed_and_preserved(con):
    con.raw_sql("CREATE TABLE t (a BIGINT, b VARCHAR)")
    con.raw_sql("INSERT INTO t VALUES (999, 'persisted')")

    t = ibis.table(dict(a="int64", b="string"), name="t")
    batch = pa.table({"a": [1, 2], "b": ["x", "y"]})

    result = con.execute(t, table_bindings={t: batch})
    assert result.a.tolist() == [1, 2]

    # the user table is still present and unchanged
    assert con.raw_sql("SELECT * FROM t").fetchall() == [(999, "persisted")]


def test_no_bindings_keeps_legacy_behavior(con):
    t = ibis.table(dict(a="int64"), name="unbound_legacy")
    with pytest.raises(duckdb.CatalogException):
        con.execute(t)
    assert con.compile(t)  # compilation itself never needed data


def test_empty_mapping_is_legacy_behavior(con):
    t = ibis.table(dict(a="int64"), name="unbound_empty")
    with pytest.raises(duckdb.CatalogException):
        con.execute(t, table_bindings={})


# ---------------------------------------------------------------------------
# entry points and result shapes
# ---------------------------------------------------------------------------


def test_all_backend_entry_points_accept_bindings(con):
    t = ibis.table(dict(a="int64", b="float64"), name="t")
    expr = t.filter(t.a > 0)
    batch = pa.table({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]})

    assert con.execute(expr, table_bindings={t: batch}).shape == (3, 2)
    out = con.to_pyarrow(expr, table_bindings={t: batch})
    assert out.num_rows == 3 and out.column_names == ["a", "b"]

    reader = con.to_pyarrow_batches(expr, table_bindings={t: batch})
    assert sum(b.num_rows for b in reader) == 3

    sql = con.compile(expr, table_bindings={t: batch})
    assert "ibis_duckdb_binding" in sql
    # compile must not have registered anything
    assert con.list_tables() == []


def test_expr_entry_points_with_explicit_backend(con):
    t = ibis.table(dict(a="int64"), name="t")
    batch = pa.table({"a": [1, 2, 3]})

    assert t.execute(backend=con, table_bindings={t: batch}).shape == (3, 1)
    assert t.to_pyarrow(backend=con, table_bindings={t: batch}).num_rows == 3
    reader = t.to_pyarrow_batches(backend=con, table_bindings={t: batch})
    assert sum(b.num_rows for b in reader) == 3
    assert "ibis_duckdb_binding" in t.compile(backend=con, table_bindings={t: batch})


def test_expr_entry_points_with_default_backend(con):
    ibis.set_backend(con)
    try:
        t = ibis.table(dict(a="int64"), name="t")
        batch = pa.table({"a": [1]})
        assert t.execute(table_bindings={t: batch}).shape == (1, 1)
    finally:
        ibis.set_backend(None)


def test_result_shapes_table_column_scalar(con):
    t = ibis.table(dict(a="int64", b="string"), name="t")
    batch = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    table = con.execute(t, table_bindings={t: batch})
    assert table.shape == (3, 2)

    column = con.execute(t.a, table_bindings={t: batch})
    assert column.tolist() == [1, 2, 3]

    scalar = con.execute(t.a.sum(), table_bindings={t: batch})
    assert scalar == 6

    pa_scalar = con.to_pyarrow(t.a.sum(), table_bindings={t: batch})
    assert pa_scalar.as_py() == 6


def test_limit_applies_with_bindings(con):
    t = ibis.table(dict(a="int64"), name="t")
    batch = pa.table({"a": list(range(10))})
    assert con.execute(t, limit=3, table_bindings={t: batch}).shape == (3, 1)


def test_scalar_params_and_bindings_together(con):
    t = ibis.table(dict(a="int64"), name="t")
    p = ibis.param("int64")
    expr = t.filter(t.a > p)
    batch = pa.table({"a": [1, 2, 3, 4]})

    assert con.execute(expr, params={p: 2}, table_bindings={t: batch}).a.tolist() == [
        3,
        4,
    ]


# ---------------------------------------------------------------------------
# query shapes
# ---------------------------------------------------------------------------


def test_cte_reuse_and_set_operations(con):
    t = ibis.table(dict(g="string", x="int64"), name="t")
    sub = t.filter(t.x > 0)
    expr = sub.union(sub.mutate(x=lambda z: z.x * 10), distinct=False).order_by("x")
    batch = pa.table({"g": ["a", "b", "a"], "x": [1, -1, 2]})

    result = con.execute(expr, table_bindings={t: batch})
    assert result.x.tolist() == [1, 2, 10, 20]


def test_grouped_aggregation(con):
    t = ibis.table(dict(g="string", x="int64"), name="t")
    expr = t.group_by("g").aggregate(cnt=t.x.count(), total=t.x.sum()).order_by("g")
    batch = pa.table({"g": ["a", "b", "a"], "x": [1, 2, 3]})

    result = con.execute(expr, table_bindings={t: batch})
    assert result.g.tolist() == ["a", "b"]
    assert result.cnt.tolist() == [2, 1]
    assert result.total.tolist() == [4, 2]


def test_window_function(con):
    t = ibis.table(dict(g="string", x="int64"), name="t")
    expr = t.mutate(
        rn=ibis.row_number().over(group_by="g", order_by=t.x.desc())
    ).order_by("g", "rn")
    batch = pa.table({"g": ["a", "a", "b"], "x": [1, 2, 3]})

    result = con.execute(expr, table_bindings={t: batch})
    assert result.rn.tolist() == [0, 1, 0]
    assert result.x.tolist() == [2, 1, 3]


def test_join_two_bindings(con):
    left = ibis.table(dict(k="int64", v="string"), name="l")
    right = ibis.table(dict(k="int64", label="string"), name="r")
    expr = (
        left.join(right, left.k == right.k)
        .select(left.k, left.v, right.label)
        .order_by("k")
    )
    dl = pa.table({"k": [1, 2], "v": ["a", "b"]})
    dr = pa.table({"k": [2, 3], "label": ["two", "three"]})

    result = con.execute(expr, table_bindings={left: dl, right: dr})
    assert result.k.tolist() == [2]
    assert result.label.tolist() == ["two"]


def test_correlated_exists(con):
    t = ibis.table(dict(g="string", x="int64"), name="t")
    d = ibis.table(dict(g="string"), name="d")
    exists = ops.ExistsSubquery(d.filter(d.g == t.g)).to_expr()
    expr = t.filter(exists).order_by("x")

    batch = pa.table({"g": ["a", "b", "a"], "x": [1, 2, 3]})
    dim = pa.table({"g": ["a"]})

    result = con.execute(expr, table_bindings={t: batch, d: dim})
    assert result.x.tolist() == [1, 3]


def test_view_self_join_shares_one_binding(con):
    t = ibis.table(dict(k="int64", v="int64"), name="t")
    v = t.view()
    # join without an equality predicate so both sides scan the same shared
    # relation independently
    expr = (
        v.cross_join(t)
        .select(v.k.name("k1"), v.v.name("lhs"), t.v.name("rhs"))
        .order_by("lhs", "rhs")
    )
    batch = pa.table({"k": [1, 2], "v": [10, 20]})

    result = con.execute(expr, table_bindings={t: batch})
    # cartesian self join over the single shared relation -> 4 rows
    assert result.shape == (4, 3)
    assert result.lhs.tolist() == [10, 10, 20, 20]
    assert result.rhs.tolist() == [10, 20, 10, 20]

    # and a keyed self join uses the same data on both sides
    keyed = (
        v.join(t, v.k == t.k)
        .select(v.k, v.v.name("lhs"), t.v.name("rhs"))
        .order_by("k")
    )
    keyed_result = con.execute(keyed, table_bindings={t: batch})
    assert keyed_result.k.tolist() == [1, 2]
    assert keyed_result.lhs.tolist() == keyed_result.rhs.tolist()


def test_same_name_different_schema_placeholders_are_distinct(con):
    ta = ibis.table(dict(a="int64"), name="same")
    tb = ibis.table(dict(a="string", b="boolean"), name="same")
    expr = ta.cross_join(tb)

    da = pa.table({"a": [7]})
    db = pa.table({"a": ["hello"], "b": [True]})

    result = con.execute(expr, table_bindings={ta: da, tb: db})
    assert result.shape == (1, 3)

    # compilation assigns two distinct relation names
    sql = con.compile(expr, table_bindings={ta: da, tb: db})
    assert sql.count("ibis_duckdb_binding") == 2


def test_structurally_equal_placeholder_uses_same_binding_semantics(con):
    t1 = ibis.table(dict(a="int64"), name="t")
    t2 = ibis.table(dict(a="int64"), name="t")
    assert t1.op() == t2.op()
    batch = pa.table({"a": [5, 6]})
    # binding via the second node feeds an expression built from the first
    assert con.execute(t1, table_bindings={t2: batch}).a.tolist() == [5, 6]


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


@pytest.fixture
def placeholder():
    return ibis.table(dict(a="int64", b="string", c="float64", d="boolean"), name="t")


def _good(table):
    return pa.table(
        {
            "a": [1],
            "b": ["x"],
            "c": [1.0],
            "d": [True],
        }
    )


def test_reject_missing_column(con, placeholder):
    data = pa.table({"a": [1], "b": ["x"], "c": [1.0]})
    with pytest.raises(IbisInputError, match=r"missing column.*d"):
        con.execute(placeholder, table_bindings={placeholder: data})


def test_reject_extra_column(con, placeholder):
    data = pa.table({"a": [1], "b": ["x"], "c": [1.0], "d": [True], "nope": [1]})
    with pytest.raises(IbisInputError, match=r"unexpected column.*nope"):
        con.execute(placeholder, table_bindings={placeholder: data})


def test_reject_wrong_arrow_type(con, placeholder):
    data = pa.table(
        {"a": pa.array([1], type=pa.int32()), "b": ["x"], "c": [1.0], "d": [True]}
    )
    with pytest.raises(IbisInputError, match=r"column 'a'.*int32"):
        con.execute(placeholder, table_bindings={placeholder: data})


def test_reject_non_arrow_value(con, placeholder):
    with pytest.raises(IbisInputError, match=r"must be a pyarrow\.Table"):
        con.execute(placeholder, table_bindings={placeholder: [1, 2, 3]})


def test_reject_non_unbound_key(con, placeholder):
    bound = con.create_table("catalog_t", _good(placeholder))
    with pytest.raises(IbisInputError, match="unbound Table expressions"):
        con.execute(bound, table_bindings={bound: _good(placeholder)})


def test_reject_foreign_object_key(con, placeholder):
    with pytest.raises(IbisInputError, match="unbound Table expressions"):
        con.execute(placeholder, table_bindings={"t": _good(placeholder)})


def test_reject_incomplete_mapping(con):
    t1 = ibis.table(dict(a="int64"), name="t1")
    t2 = ibis.table(dict(b="int64"), name="t2")
    expr = t1.cross_join(t2)
    with pytest.raises(IbisInputError, match=r"missing binding.*t2"):
        con.execute(expr, table_bindings={t1: pa.table({"a": [1]})})


def test_reject_extra_key(con, placeholder):
    other = ibis.table(dict(z="int64"), name="other")
    with pytest.raises(IbisInputError, match="do not appear in the expression"):
        con.execute(
            placeholder,
            table_bindings={
                placeholder: _good(placeholder),
                other: pa.table({"z": [1]}),
            },
        )


def test_reject_non_nullable_violation(con):
    t = ibis.table(dict(a="int64", b="!float64"), name="t")
    data = pa.table({"a": [1, 2], "b": [1.0, None]})
    with pytest.raises(IbisInputError, match=r"non-nullable.*null value"):
        con.execute(t, table_bindings={t: data})


def test_nullable_flag_difference_is_accepted(con):
    t = ibis.table(dict(a="int64"), name="t")
    data = pa.table(
        [pa.array([1, 2], type=pa.int64())],
        schema=pa.schema([pa.field("a", pa.int64(), nullable=False)]),
    )
    assert con.execute(t, table_bindings={t: data}).shape == (2, 1)


def test_supported_timestamp_microsecond_no_timezone(con):
    t = ibis.table(dict(ts="timestamp"), name="t")
    good = pa.table({"ts": pa.array([], type=pa.timestamp("us"))})
    assert con.execute(t, table_bindings={t: good}).empty

    bad_unit = pa.table({"ts": pa.array([], type=pa.timestamp("ns"))})
    with pytest.raises(IbisInputError, match="timestamp"):
        con.execute(t, table_bindings={t: bad_unit})

    bad_tz = pa.table({"ts": pa.array([], type=pa.timestamp("us", tz="UTC"))})
    with pytest.raises(IbisInputError, match="timestamp"):
        con.execute(t, table_bindings={t: bad_tz})


def test_unsupported_ibis_type_rejected(con):
    t = ibis.table(dict(d="date"), name="t")
    data = pa.table({"d": pa.array([], type=pa.date32())})
    with pytest.raises(IbisInputError, match="unsupported type"):
        con.execute(t, table_bindings={t: data})


def test_empty_table_requires_full_structure(con, placeholder):
    empty_full = pa.table(
        {
            "a": pa.array([], type=pa.int64()),
            "b": pa.array([], type=pa.string()),
            "c": pa.array([], type=pa.float64()),
            "d": pa.array([], type=pa.bool_()),
        }
    )
    result = con.execute(placeholder, table_bindings={placeholder: empty_full})
    assert result.empty

    empty_missing = pa.table({"a": pa.array([], type=pa.int64())})
    with pytest.raises(IbisInputError, match=r"missing column"):
        con.execute(placeholder, table_bindings={placeholder: empty_missing})


def test_column_reorder_accepted_and_input_untouched(con, placeholder):
    reordered = pa.table({"d": [True], "c": [1.0], "b": ["x"], "a": [1]})
    original_order = reordered.column_names
    result = con.execute(placeholder, table_bindings={placeholder: reordered})
    assert result.columns.tolist() == ["a", "b", "c", "d"]
    assert result.a.tolist() == [1]
    # caller's arrow table is not mutated
    assert reordered.column_names == original_order


def test_validation_happens_before_registration(con, placeholder):
    bad = pa.table({"a": pa.array([1], type=pa.int32())})
    with pytest.raises(IbisInputError):
        con.execute(placeholder, table_bindings={placeholder: bad})
    assert con.list_tables() == []


def test_expression_and_mapping_are_not_mutated(con, placeholder):
    data = _good(placeholder)
    before = repr(placeholder.op())
    mapping = {placeholder: data}
    con.execute(placeholder, table_bindings=mapping)
    assert repr(placeholder.op()) == before
    assert isinstance(placeholder.op(), ops.UnboundTable)
    assert set(mapping) == {placeholder}
    assert mapping[placeholder] is data


# ---------------------------------------------------------------------------
# cross-backend database tables
# ---------------------------------------------------------------------------


def test_database_table_must_come_from_executing_backend(con):
    other = ibis.duckdb.connect()
    try:
        other.raw_sql("CREATE TABLE foreign_t (a BIGINT)")
        foreign = other.table("foreign_t")
        local = ibis.table(dict(a="int64"), name="local")
        expr = local.cross_join(foreign)
        data = pa.table({"a": [1]})
        with pytest.raises(IbisError, match=r"different.*backend"):
            con.execute(expr, table_bindings={local: data})
        with pytest.raises(IbisError, match=r"different.*backend"):
            con.compile(expr, table_bindings={local: data})
    finally:
        other.disconnect()


def test_mixed_bound_and_unbound_graph(con):
    con.raw_sql("CREATE TABLE dim (g VARCHAR, label VARCHAR)")
    con.raw_sql("INSERT INTO dim VALUES ('a', 'alpha')")
    dim = con.table("dim")

    fact = ibis.table(dict(g="string", x="int64"), name="fact")
    expr = fact.join(dim, fact.g == dim.g).select(fact.x, dim.label)
    data = pa.table({"g": ["a"], "x": [42]})

    result = con.execute(expr, table_bindings={fact: data})
    assert result.label.tolist() == ["alpha"]


# ---------------------------------------------------------------------------
# compile/execute consistency
# ---------------------------------------------------------------------------


def test_compile_does_not_register_or_read(con):
    t = ibis.table(dict(a="int64"), name="t")
    data = pa.table({"a": [1, 2, 3]})
    sql = con.compile(t, table_bindings={t: data})
    assert con.list_tables() == []
    # relation names derive from the placeholder's structural identity, so
    # compiling the same mapping twice yields byte-identical SQL
    execute_sql = con.compile(t, table_bindings={t: data})
    assert sql == execute_sql
    assert con.execute(t, table_bindings={t: data}).a.tolist() == [1, 2, 3]


def test_compile_against_unbound_without_binding_works(con):
    t = ibis.table(dict(a="int64"), name="t")
    sql = con.compile(t)
    assert '"t"' in sql


# ---------------------------------------------------------------------------
# lifecycle / resource management
# ---------------------------------------------------------------------------


def test_eager_call_leans_up_registrations(con):
    t = ibis.table(dict(a="int64"), name="t")
    data = pa.table({"a": [1, 2]})
    for _ in range(5):
        con.execute(t, table_bindings={t: data})
    assert con.list_tables() == []


def test_failed_sql_execution_leaves_nothing_and_reuse_works(con):
    t = ibis.table(dict(a="int64"), name="t")
    data = pa.table({"a": [1, 0, 2]})

    @udf.scalar.python
    def boom(x: int) -> int:
        if x == 0:
            raise RuntimeError("boom")
        return x

    bad_expr = t.mutate(x=boom(t.a))
    with pytest.raises(duckdb.InvalidInputException):
        con.execute(bad_expr, table_bindings={t: data})

    assert con.list_tables() == []
    # a subsequent correct call on the same expression/backend works
    assert con.execute(t, table_bindings={t: data}).shape == (3, 1)


def test_interleaved_streaming_readers_and_eager_query(con):
    t = ibis.table(dict(a="int64"), name="t")
    d1 = pa.table({"a": range(100_000)})
    d2 = pa.table({"a": range(100_000, 200_000)})
    d3 = pa.table({"a": range(1_000_000, 1_001_000)})

    r1 = con.to_pyarrow_batches(t, table_bindings={t: d1}, chunk_size=1024)
    r2 = con.to_pyarrow_batches(t, table_bindings={t: d2}, chunk_size=1024)

    first1 = r1.read_next_batch()
    first2 = r2.read_next_batch()
    assert first1.column(0)[0].as_py() == 0
    assert first2.column(0)[0].as_py() == 100_000

    # third eager binding interleaved on the same backend
    eager = con.execute(t, table_bindings={t: d3})
    assert len(eager) == 1000

    # neither cursor was clobbered
    next1 = r1.read_next_batch()
    next2 = r2.read_next_batch()
    assert next1.column(0)[0].as_py() == 1024
    assert next2.column(0)[0].as_py() == 101_024

    r1.close()
    # every row of r2 must still be readable: the two batches already
    # pulled plus whatever the iterator yields
    pulled_r2 = first2.num_rows + next2.num_rows
    remaining = pulled_r2 + sum(b.num_rows for b in r2)
    assert remaining == 100_000
    assert con.list_tables() == []


def test_streaming_error_releases_resources(con):
    t = ibis.table(dict(a="int64"), name="t")
    data = pa.table({"a": range(100_000)})

    @udf.scalar.python
    def boom(x: int) -> int:
        if x > 5:
            raise RuntimeError("boom")
        return x

    with pytest.raises(duckdb.InvalidInputException):
        reader = con.to_pyarrow_batches(
            t.mutate(x=boom(t.a)), table_bindings={t: data}, chunk_size=2
        )
        reader.read_all()

    assert con.list_tables() == []
    gc.collect()
    assert con.list_tables() == []


def test_reader_exhaustion_closes_session(con):
    t = ibis.table(dict(a="int64"), name="t")
    data = pa.table({"a": range(5_000)})
    reader = con.to_pyarrow_batches(t, table_bindings={t: data}, chunk_size=512)
    assert sum(b.num_rows for b in reader) == 5_000
    assert con.list_tables() == []


def test_many_round_trips_return_to_baseline(con):
    t = ibis.table(dict(a="int64", b="string"), name="t")
    data = pa.table({"a": list(range(100)), "b": [str(i) for i in range(100)]})
    for _ in range(200):
        con.execute(t, table_bindings={t: data})
        reader = con.to_pyarrow_batches(t, table_bindings={t: data})
        sum(b.num_rows for b in reader)
    assert con.list_tables() == []
    gc.collect()
    # backend itself stays open and usable
    assert con.execute(t, table_bindings={t: data}).shape == (100, 2)


def test_session_introspection_properties(con):
    t = ibis.table(dict(a="int64"), name="t")
    data = pa.table({"a": [1]})
    session = BindingSession(con, t, {t: data})
    assert session.holds_data
    assert len(session.registered_names) == 0
    session.register_all(con.con)
    assert len(session.registered_names) == 1
    session.close()
    assert not session.holds_data
    assert con.list_tables() == []


def test_backend_connection_is_not_closed_by_session(con):
    t = ibis.table(dict(a="int64"), name="t")
    data = pa.table({"a": [1]})
    con.execute(t, table_bindings={t: data})
    reader = con.to_pyarrow_batches(t, table_bindings={t: data})
    reader.close()
    # backend still works for ordinary operations
    assert con.raw_sql("SELECT 1").fetchone() == (1,)
