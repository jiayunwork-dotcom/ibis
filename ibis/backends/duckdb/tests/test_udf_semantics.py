"""Regression tests for DuckDB scalar UDF configuration and evaluation semantics.

These tests pin down three independent failures that used to occur when a
``@ibis.udf.scalar.python`` / ``@ibis.udf.scalar.pyarrow`` decorated function
was executed through the DuckDB backend:

1. decorator options (``null_handling``, ``exception_handling``,
   ``side_effects``) were accepted by Ibis but never forwarded to
   ``duckdb.DuckDBPyConnection.create_function``, so they had no effect at
   runtime;
2. a UDF declared with ``side_effects=True`` was compiled like a pure value,
   letting the relational rewrites duplicate or collapse its call sites;
3. Python callbacks must stay on the database side, and streaming/reader paths
   must not buffer a whole result just to reproduce the semantics.

The tests use a plain Python counter (and, for the Arrow flavour, a per-batch
counter) plus direct ``duckdb`` invocations as the reference. No test asserts
row/column evaluation order; only call counts, uniqueness of side-effect values
and the per-row ``b == a + 1`` relation are checked.
"""

from __future__ import annotations

import duckdb
import pytest

import ibis
from ibis import udf

pa = pytest.importorskip("pyarrow")
pc = pytest.importorskip("pyarrow.compute")


@pytest.fixture
def con():
    return ibis.duckdb.connect()


@pytest.fixture
def details(con):
    """Three-row integer detail table usable across the whole file."""
    return ibis.memtable({"x": [1, 2, 3]}, name="details")


class Counter:
    """A shared, explicit call counter for side-effecting UDFs."""

    def __init__(self) -> None:
        self.inputs: list[object] = []
        self.results: list[int] = []

    def bump(self, value: object) -> int:
        self.inputs.append(value)
        result = len(self.inputs)
        self.results.append(result)
        return result

    @property
    def calls(self) -> int:
        return len(self.inputs)

    def reset(self) -> None:
        self.inputs.clear()
        self.results.clear()


# ---------------------------------------------------------------------------
# reference semantics straight from duckdb
# ---------------------------------------------------------------------------


def test_reference_direct_duckdb_side_effects():
    """Sanity-check the reference behaviour the ibis tests compare against."""
    con = duckdb.connect()
    calls: list[int] = []

    def f(x):
        calls.append(x)
        return len(calls)

    con.create_function("f_ref", f, ["BIGINT"], "BIGINT", side_effects=True)
    con.execute("CREATE TABLE tt AS SELECT * FROM range(3) t(x)")

    dups = con.execute("SELECT f_ref(x) AS a, f_ref(x) AS b FROM tt").fetchall()
    assert len(calls) == 6
    assert all(a != b for a, b in dups)

    calls.clear()
    named = con.execute(
        "SELECT a, a + 1 AS b FROM (SELECT f_ref(x) AS a FROM tt) s"
    ).fetchall()
    assert len(calls) == 3
    assert all(b == a + 1 for a, b in named)


# ---------------------------------------------------------------------------
# null_handling / exception_handling
# ---------------------------------------------------------------------------


def test_null_handling_special_handles_null(con):
    @udf.scalar.python(null_handling="special")
    def spy(x: int) -> int:
        return 77 if x is None else x

    assert con.execute(spy(ibis.null().cast("int64"))) == 77

    t = ibis.memtable({"x": [None, 2, 3]}).mutate(x=lambda t: t.x.cast("int64"))
    assert con.execute(spy(t.x)).tolist() == [77, 2, 3]
    assert con.to_pyarrow(spy(t.x)).to_pylist() == [77, 2, 3]

    table = con.execute(t.select(y=spy(t.x)))
    assert table.y.tolist() == [77, 2, 3]
    batches = con.to_pyarrow_batches(t.select(y=spy(t.x)))
    assert [b.column("y").to_pylist() for b in batches] == [[77, 2, 3]]


def test_default_null_handling_does_not_intercept(con):
    """Without ``special`` the callback must not see NULL rows at all."""

    seen = []

    @udf.scalar.python
    def spy(x: int) -> int:
        seen.append(x)
        return x

    t = ibis.memtable({"x": [None, 2, 3]}).mutate(x=lambda t: t.x.cast("int64"))
    result = con.execute(spy(t.x))
    assert result.isna().tolist() == [True, False, False]
    assert result.fillna(-1).tolist() == [-1, 2, 3]
    assert None not in seen
    assert seen == [2, 3]


def test_exception_handling_return_null(con):
    @udf.scalar.python(exception_handling="return_null")
    def boom(x: int) -> int:
        raise ValueError("boom")

    assert con.execute(boom(ibis.literal(1))) is None

    t = ibis.memtable({"x": [1, 2, 3]})
    assert con.execute(t.select(y=boom(t.x))).y.tolist() == [None, None, None]


def test_mixed_null_and_non_null_inputs(con):
    """A single column mixing NULL and non-NULL values, with `special`."""

    @udf.scalar.python(null_handling="special")
    def keep_or_77(x: int) -> int:
        return 77 if x is None else x * 10

    t = ibis.memtable({"x": [None, 1, None, 4]}).mutate(x=lambda t: t.x.cast("int64"))
    expected = [77, 10, 77, 40]

    assert con.execute(keep_or_77(t.x)).tolist() == expected
    assert con.to_pyarrow(keep_or_77(t.x)).to_pylist() == expected
    assert con.execute(t.select(y=keep_or_77(t.x))).y.tolist() == expected
    batches = con.to_pyarrow_batches(t.select(y=keep_or_77(t.x)))
    assert [v for b in batches for v in b.column("y").to_pylist()] == expected


def test_exception_conversion_only_affects_own_function(con):
    """``return_null`` must not leak into a sibling function's config."""

    @udf.scalar.python(exception_handling="return_null")
    def maybe_boom(x: int) -> int:
        if x == 2:
            raise ValueError("boom")
        return x

    @udf.scalar.python
    def always_boom(x: int) -> int:
        raise ValueError("boom")

    t = ibis.memtable({"x": [1, 2, 3]})
    assert con.to_pyarrow(maybe_boom(t.x)).to_pylist() == [1, None, 3]
    with pytest.raises(Exception, match="boom"):
        con.execute(always_boom(t.x))


def test_default_exception_propagation_then_reuse(con):
    # a persistent user table exists *before* the failing query
    con.raw_sql("CREATE TABLE user_t (a BIGINT)")
    con.raw_sql("INSERT INTO user_t VALUES (10), (20)")

    @udf.scalar.python
    def boom(x: int) -> int:
        raise ValueError("boom")

    try:
        con.execute(boom(ibis.literal(1)))
    except Exception as exc:  # noqa: BLE001 - duckdb raises its own error type
        assert "boom" in str(exc)
    else:  # pragma: no cover - the callback must propagate
        raise AssertionError("expected the Python exception to propagate")

    # the backend, its catalog and the pre-existing user table survive
    @udf.scalar.python
    def add_one(x: int) -> int:
        return x + 1

    assert con.execute(add_one(ibis.literal(1))) == 2
    assert con.execute(con.table("user_t")).a.tolist() == [10, 20]
    assert con.list_tables() == ["user_t"]


def test_same_python_name_different_config(con):
    """Two distinct decorated functions sharing ``__name__`` must not collide."""

    def make(ret, **config):
        @udf.scalar.python(name="shared", **config)
        def shared(x: int) -> int:
            return ret if x is not None else -1

        return shared

    f1 = make(100)
    f2 = make(200, null_handling="special")

    t = ibis.memtable({"x": [None, 5]}).mutate(x=lambda t: t.x.cast("int64"))
    expr = t.select(a=f1(t.x), b=f2(t.x))
    result = con.to_pyarrow(expr)
    # f1 uses DuckDB's default NULL interception: it never sees NULL and that
    # row stays NULL.  f2 opted into ``special`` and handles NULL itself.
    assert result.column("a").to_pylist() == [None, 100]
    assert result.column("b").to_pylist() == [-1, 200]

    # re-execution is stable and both functions keep their own definition
    assert con.to_pyarrow(expr).to_pydict() == result.to_pydict()


# ---------------------------------------------------------------------------
# side_effects boundary
# ---------------------------------------------------------------------------


def test_side_effects_direct_selection_evaluates_each_site(con, details):
    counter = Counter()

    @udf.scalar.python(side_effects=True)
    def f(x: int) -> int:
        return counter.bump(x)

    result = con.execute(details.select(a=f(details.x), b=f(details.x)))

    assert counter.calls == 6
    # each of the six calls returns a distinct number: nothing was merged
    assert sorted(counter.results) == [1, 2, 3, 4, 5, 6]
    assert len(result) == 3
    assert set(result.a) | set(result.b) == set(range(1, 7))
    assert (result.a != result.b).all()


def test_side_effects_named_result_is_reused(con, details):
    counter = Counter()

    @udf.scalar.python(side_effects=True)
    def f(x: int) -> int:
        return counter.bump(x)

    m = details.mutate(z=f(details.x))
    result = con.execute(m.select(a=m.z, b=m.z + 1))

    assert counter.calls == 3
    assert (result.b == result.a + 1).all()
    assert sorted(result.a) == [1, 2, 3]
    assert len(set(result.a)) == 3


def test_side_effects_named_filter_and_projection(con, details):
    counter = Counter()

    @udf.scalar.python(side_effects=True)
    def f(x: int) -> int:
        return counter.bump(x)

    m = details.mutate(z=f(details.x))

    filtered = con.execute(m.filter(m.z > 1))
    assert counter.calls == 3
    assert sorted(filtered.z) == [2, 3]

    counter.reset()
    projected = con.execute(m.filter(m.z > 1).select(m.z, w=m.z + 1))
    assert counter.calls == 3
    assert (projected.w == projected.z + 1).all()

    counter.reset()
    renamed = m.rename(w="z")
    out = con.execute(renamed.select(a=renamed.w, b=renamed.w + 1))
    assert counter.calls == 3
    assert (out.b == out.a + 1).all()

    counter.reset()
    out = con.execute(renamed.filter(renamed.w > 1))
    assert counter.calls == 3
    assert sorted(out.w) == [2, 3]


def test_side_effects_compile_does_not_run(con, details):
    counter = Counter()

    @udf.scalar.python(side_effects=True)
    def f(x: int) -> int:
        return counter.bump(x)

    m = details.mutate(z=f(details.x))
    sql = con.compile(m.select(a=m.z, b=m.z + 1))
    assert counter.calls == 0
    # the named value is computed once in a subquery and referenced by name
    assert sql.count(type(f(details.x).op()).__name__.upper()) == 1


def test_side_effects_repeated_execution_is_deterministic(con, details):
    counter = Counter()

    @udf.scalar.python(side_effects=True)
    def f(x: int) -> int:
        return counter.bump(x)

    m = details.mutate(z=f(details.x))
    expr = m.select(a=m.z, b=m.z + 1)

    first = con.execute(expr)
    assert counter.calls == 3
    assert (first.b == first.a + 1).all()

    second = con.execute(expr)
    # call count is per execution; the per-row relation still holds
    assert counter.calls == 6
    assert (second.b == second.a + 1).all()


def test_side_effects_project_after_project(con, details):
    """Two chained projections must not inline the named result twice."""
    counter = Counter()

    @udf.scalar.python(side_effects=True)
    def f(x: int) -> int:
        return counter.bump(x)

    inner = details.mutate(z=f(details.x))
    outer = inner.select(z=inner.z)
    result = con.execute(outer.select(a=outer.z, b=outer.z + 1))

    assert counter.calls == 3
    assert (result.b == result.a + 1).all()
    assert sorted(result.a) == [1, 2, 3]


def test_side_effects_multi_batch_python(con):
    """A per-row Python UDF still streams: one call per row, many batches."""
    n = 50_000
    calls: list[int] = []

    @udf.scalar.python(side_effects=True)
    def f(x: int) -> int:
        calls.append(x)
        return x + 1

    t = ibis.memtable({"x": list(range(n))})
    reader = con.to_pyarrow_batches(t.select(y=f(t.x)), chunk_size=4096)

    total = 0
    batch_count = 0
    for batch in reader:
        batch_count += 1
        total += batch.num_rows
    assert total == n
    assert batch_count > 1
    # Python UDFs are row-at-a-time: exactly one call per row
    assert len(calls) == n


def test_pure_udf_still_optimized(con, details):
    counter = Counter()

    @udf.scalar.python
    def pure(x: int) -> int:
        counter.bump(x)
        return x * 2

    # pure values can be inlined: the two references share one evaluation,
    # which is the safe optimisation for a deterministic function
    m = details.mutate(z=pure(details.x))
    result = con.execute(m.select(a=m.z, b=m.z + 1))
    assert (result.b == result.a + 1).all()
    assert counter.calls == 3

    # and the direct two-site selection is common-subexpression-eliminated
    counter.reset()
    direct = con.execute(details.select(a=pure(details.x), b=pure(details.x)))
    assert counter.calls == 3
    assert (direct.a == direct.b).all()


# ---------------------------------------------------------------------------
# Arrow UDFs: config transport and per-batch semantics
# ---------------------------------------------------------------------------


def test_arrow_null_handling_special(con):
    batches: list[int] = []

    @udf.scalar.pyarrow(null_handling="special")
    def f(a: int) -> int:
        batches.append(len(a))
        return pa.array(
            [77 if v is None else v for v in a.to_pylist()], type=pa.int64()
        )

    t = ibis.memtable({"x": [None, 2, 3]}).mutate(x=lambda t: t.x.cast("int64"))
    assert con.execute(f(t.x)).tolist() == [77, 2, 3]
    assert con.to_pyarrow(f(t.x)).to_pylist() == [77, 2, 3]
    # the callback is vectorized: at least one call, never one per row
    assert sum(batches) >= 1
    assert len(batches) < 3

    batches.clear()
    out = [
        v
        for b in con.to_pyarrow_batches(t.select(y=f(t.x)))
        for v in b.column("y").to_pylist()
    ]
    assert out == [77, 2, 3]
    assert sum(batches) >= 1
    assert len(batches) < 3


def test_arrow_exception_handling_return_null(con):
    @udf.scalar.pyarrow(exception_handling="return_null")
    def boom(a: int) -> int:
        raise ValueError("boom")

    t = ibis.memtable({"x": [1, 2, 3]})
    assert con.execute(boom(t.x)).tolist() == [None, None, None]
    assert con.to_pyarrow(boom(t.x)).to_pylist() == [None, None, None]
    batches = con.to_pyarrow_batches(boom(t.x).as_table())
    assert [v for b in batches for v in b.column(0).to_pylist()] == [None, None, None]


def test_arrow_default_exception_propagates(con):
    @udf.scalar.pyarrow
    def boom(a: int) -> int:
        raise ValueError("boom")

    t = ibis.memtable({"x": [1, 2, 3]})
    with pytest.raises(Exception, match="boom"):
        con.execute(boom(t.x))
    # still usable afterwards
    assert con.execute(ibis.literal(1)) == 1


def test_arrow_side_effects_per_batch_not_per_row(con):
    n = 200_000
    calls: list[int] = []

    @udf.scalar.pyarrow(side_effects=True)
    def bump(a: int) -> int:
        calls.append(len(a))
        return pc.add(a, 1)

    t = ibis.memtable({"x": list(range(n))})
    expr = t.select(y=bump(t.x))

    assert con.compile(expr)
    assert calls == []

    result = con.execute(expr)
    assert result.y.tolist()[:3] == [1, 2, 3]
    assert sum(calls) == n
    # Arrow UDFs are called per batch, not per row
    assert len(calls) < n


def test_arrow_side_effects_batches_are_streamed(con):
    n = 200_000
    calls: list[int] = []

    @udf.scalar.pyarrow(side_effects=True)
    def bump(a: int) -> int:
        calls.append(len(a))
        return pc.add(a, 1)

    t = ibis.memtable({"x": list(range(n))})
    expr = t.select(y=bump(t.x))

    reader = con.to_pyarrow_batches(expr, chunk_size=4096)
    # bounded prefix may be pulled already; never the whole 200k rows
    assert 0 < sum(calls) < n

    total = 0
    batch_count = 0
    for batch in reader:
        batch_count += 1
        total += batch.num_rows
        # the first batches are visible before the whole input is consumed
        if batch_count == 1:
            first_seen_calls = sum(calls)
    assert total == n
    assert batch_count > 1
    assert sum(calls) == n
    assert len(calls) > batch_count  # still per batch, not per row
    assert first_seen_calls < n


# ---------------------------------------------------------------------------
# streaming, compile and relation bindings
# ---------------------------------------------------------------------------


def test_compile_never_runs_callback(con):
    calls: list[int] = []

    @udf.scalar.python(side_effects=True)
    def f(x: int) -> int:
        calls.append(x)
        return x

    t = ibis.table({"x": "int64"}, name="t")
    expr = t.mutate(z=f(t.x)).select(a=lambda q: q.z, b=lambda q: q.z + 1)
    con.compile(expr)
    assert calls == []


def test_streaming_does_not_materialize_whole_result(con):
    """A wide-ish artificial detail set is consumed batch by batch."""
    n = 200_000
    # a plain integer counter: the test never retains the input keys or any
    # output rows, so nothing here grows with the result size
    calls = [0]

    @udf.scalar.python(side_effects=True)
    def f(x: int) -> str:
        calls[0] += 1
        return "a" * 200

    t = ibis.memtable({"x": list(range(n))})
    expr = t.select(y=f(t.x))

    sql = con.compile(expr)
    assert calls[0] == 0

    reader = con.to_pyarrow_batches(expr, chunk_size=4096)
    # a bounded prefix may already be pulled by the reader; never the whole set
    assert 0 < calls[0] < n
    # a full result would be at least n * 200 bytes of payload
    full_bytes = n * 200

    first = reader.read_next_batch()
    assert first.num_rows > 0
    # each returned batch is a small slice, far below the whole payload
    assert first.nbytes < full_bytes // 2

    calls_after_first = calls[0]
    assert 0 < calls_after_first < n

    # draining advances the counter batch by batch without holding the result
    drained = first.num_rows
    max_batch_bytes = first.nbytes
    for batch in reader:
        drained += batch.num_rows
        max_batch_bytes = max(max_batch_bytes, batch.nbytes)
    assert drained == n
    assert calls[0] == n
    # no single batch buffers anything close to the full 40 MB output
    assert max_batch_bytes < full_bytes // 2
    assert sql  # compile produced a plan even though we never executed it


def test_side_effects_with_table_bindings(con):
    t = ibis.table({"x": "int64"}, name="t")
    counter = Counter()

    @udf.scalar.python(side_effects=True)
    def f(x: int) -> int:
        return counter.bump(x)

    m = t.mutate(z=f(t.x))
    expr = m.select(a=m.z, b=m.z + 1)

    data = pa.table({"x": [1, 2, 3, 4]})
    result = con.execute(expr, table_bindings={t: data})
    assert counter.calls == 4
    assert (result.b == result.a + 1).all()
    assert sorted(result.a) == [1, 2, 3, 4]

    counter.reset()
    reader = con.to_pyarrow_batches(expr, table_bindings={t: data}, chunk_size=2)
    rows = [b for batch in reader for b in batch.column("a").to_pylist()]
    assert len(rows) == 4
    assert counter.calls == 4
    # bindings are cleaned up after the reader is exhausted
    assert con.list_tables() == []


def test_side_effects_with_database_table(con):
    """A catalog table is an ordinary input source for a side-effecting UDF."""
    con.raw_sql("CREATE TABLE details (x BIGINT)")
    con.raw_sql("INSERT INTO details VALUES (1), (2), (3)")

    counter = Counter()

    @udf.scalar.python(side_effects=True)
    def f(x: int) -> int:
        return counter.bump(x)

    d = con.table("details")
    m = d.mutate(z=f(d.x))
    result = con.execute(m.select(a=m.z, b=m.z + 1))
    assert counter.calls == 3
    assert (result.b == result.a + 1).all()
    assert sorted(result.a) == [1, 2, 3]
    # the input table is untouched
    assert con.execute(d).x.tolist() == [1, 2, 3]


def test_side_effects_duplicate_calls_with_bindings(con):
    t = ibis.table({"x": "int64"}, name="t")
    counter = Counter()

    @udf.scalar.python(side_effects=True)
    def f(x: int) -> int:
        return counter.bump(x)

    data = pa.table({"x": [1, 2]})
    result = con.execute(t.select(a=f(t.x), b=f(t.x)), table_bindings={t: data})
    assert counter.calls == 4
    assert (result.a != result.b).all()


# ---------------------------------------------------------------------------
# shape consistency across entry points
# ---------------------------------------------------------------------------


def test_result_shapes_consistent(con):
    @udf.scalar.python
    def add_one(x: int) -> int:
        return x + 1

    t = ibis.memtable({"x": [1, 2, 3]})

    scalar = add_one(ibis.literal(1))
    assert con.execute(scalar) == 2
    assert con.to_pyarrow(scalar).as_py() == 2

    column = add_one(t.x)
    assert con.execute(column).tolist() == [2, 3, 4]
    assert con.to_pyarrow(column).to_pylist() == [2, 3, 4]

    table = t.select(y=add_one(t.x))
    assert con.execute(table).y.tolist() == [2, 3, 4]
    assert con.to_pyarrow(table).column("y").to_pylist() == [2, 3, 4]
    batches = list(con.to_pyarrow_batches(table))
    assert [v for b in batches for v in b.column("y").to_pylist()] == [2, 3, 4]
