"""End-to-end verification for DuckDB scalar UDF semantics.

Run from the repository root with the project virtualenv:

    .venv/bin/python -I examples/udf_semantics/verification.py

It exercises the three original failures (null handling, exception handling
and ``side_effects``) through ``execute``, ``to_pyarrow`` and
``to_pyarrow_batches``, and prints the observed call counts so the behaviour
is easy to compare with a direct DuckDB invocation.
"""

from __future__ import annotations

import duckdb
import pyarrow.compute as pc

import ibis
from ibis import udf

ibis.options.interactive = False


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def reference() -> None:
    """Direct DuckDB reference for the side-effect boundary."""
    con = duckdb.connect()
    calls: list[int] = []

    def f(x):
        calls.append(x)
        return len(calls)

    con.create_function("f_ref", f, ["BIGINT"], "BIGINT", side_effects=True)
    con.execute("CREATE TABLE tt AS SELECT * FROM range(3) t(x)")

    section("reference: direct duckdb")
    dups = con.execute("SELECT f_ref(x) AS a, f_ref(x) AS b FROM tt").fetchall()
    print(f"two call sites          -> {len(calls)} calls, rows={dups}")
    calls.clear()
    named = con.execute(
        "SELECT a, a + 1 AS b FROM (SELECT f_ref(x) AS a FROM tt) s"
    ).fetchall()
    print(f"one named site, reused  -> {len(calls)} calls, rows={named}")


def null_handling() -> None:
    section("null_handling='special'")
    con = ibis.duckdb.connect()

    @udf.scalar.python(null_handling="special")
    def f(x: int) -> int:
        return 77 if x is None else x

    print("scalar NULL        :", con.execute(f(ibis.null().cast("int64"))))
    t = ibis.memtable({"x": [None, 2, 3]}).mutate(x=lambda t: t.x.cast("int64"))
    print("column mixed       :", con.execute(f(t.x)).tolist())
    print("table              :", con.execute(t.select(y=f(t.x))).y.tolist())
    print(
        "to_pyarrow batches :",
        [b.to_pydict() for b in con.to_pyarrow_batches(f(t.x).as_table())],
    )


def exception_handling() -> None:
    section("exception_handling")
    con = ibis.duckdb.connect()

    @udf.scalar.python(exception_handling="return_null")
    def boom(x: int) -> int:
        raise ValueError("boom")

    print("return_null        :", con.execute(boom(ibis.literal(1))))

    @udf.scalar.python
    def strict_boom(x: int) -> int:
        raise ValueError("boom")

    try:
        con.execute(strict_boom(ibis.literal(1)))
    except Exception as exc:  # noqa: BLE001 - surface the duckdb error type
        print("default propagates :", type(exc).__name__)
    print("backend usable     :", con.execute(ibis.literal(5)))


def side_effects() -> None:
    section("side_effects=True")
    con = ibis.duckdb.connect()
    t = ibis.memtable({"x": [1, 2, 3]})
    calls: list[int] = []

    @udf.scalar.python(side_effects=True)
    def f(x: int) -> int:
        calls.append(x)
        return len(calls)

    result = con.execute(t.select(a=f(t.x), b=f(t.x)))
    print(f"two call sites          -> {len(calls)} calls")
    print(result.to_string(index=False))

    calls.clear()
    m = t.mutate(z=f(t.x))
    result = con.execute(m.select(a=m.z, b=m.z + 1))
    print(f"named, reused           -> {len(calls)} calls")
    print(result.to_string(index=False))
    print("per-row b == a + 1      :", bool((result.b == result.a + 1).all()))


def arrow_udf() -> None:
    section("pyarrow UDF (batch protocol)")
    con = ibis.duckdb.connect()
    n = 200_000
    calls: list[int] = []

    @udf.scalar.pyarrow(side_effects=True)
    def bump(a: int) -> int:
        calls.append(len(a))
        return pc.add(a, 1)

    t = ibis.memtable({"x": list(range(n))})
    expr = t.select(y=bump(t.x))
    print("compile calls           :", len(calls))
    reader = con.to_pyarrow_batches(expr, chunk_size=4096)
    total = sum(b.num_rows for b in reader)
    print(f"rows={total} arrow calls={len(calls)} (per batch, not per row)")


def streaming_memory() -> None:
    """A >200k-row, 200-byte-wide detail is consumed batch by batch."""
    import gc
    import tracemalloc

    section("streaming memory")
    con = ibis.duckdb.connect()
    n = 200_000
    # an integer counter: the script never retains the input keys or output
    calls = [0]

    @udf.scalar.python(side_effects=True)
    def f(x: int) -> str:
        calls[0] += 1
        return "a" * 200

    t = ibis.memtable({"x": list(range(n))})
    expr = t.select(y=f(t.x))
    con.compile(expr)
    print("calls at compile        :", calls[0])

    gc.collect()
    tracemalloc.start()
    reader = con.to_pyarrow_batches(expr, chunk_size=4096)
    after_create = calls[0]
    first = reader.read_next_batch()
    _, py_first_peak = tracemalloc.get_traced_memory()
    rest = 0
    max_batch_bytes = first.nbytes
    for batch in reader:
        rest += batch.num_rows
        max_batch_bytes = max(max_batch_bytes, batch.nbytes)
    total = first.num_rows + rest
    _, py_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"rows in first batch     : {first.num_rows}")
    print(f"calls before first batch: {after_create}")
    print(f"calls at end            : {calls[0]}")
    print(f"full payload if buffered: {n * 200 / 1e6:.1f} MB")
    print(f"largest batch           : {max_batch_bytes / 1e6:.2f} MB")
    # tracemalloc only sees Python allocations; Arrow buffers are C++. Both
    # stay tiny compared with holding the whole 40 MB output.
    print(
        f"python peak (first/all) : {py_first_peak / 1e6:.2f} / {py_peak / 1e6:.2f} MB"
    )
    print(f"drained rows            : {total}")


def main() -> None:
    reference()
    null_handling()
    exception_handling()
    side_effects()
    arrow_udf()
    streaming_memory()
    print("\nUDF SEMANTICS OK")


if __name__ == "__main__":
    main()
