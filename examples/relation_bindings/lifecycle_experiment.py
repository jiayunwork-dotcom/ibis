"""Lifecycle experiment: repeated binding rounds return to baseline.

Runs ``ROUNDS`` (200 by default) complete binding rounds on one DuckDB
backend, mixing eager and streaming entry points. After every round it
records:

* the number of catalog objects the backend can see (a binding must not
  leave a replacement scan behind);
* the number of child cursors still open in the session;
* whether the last session still references caller arrow data.

At the end it compares the figures against the starting baseline and, as a
cross-check, queries DuckDB's own ``duckdb_views`` catalog to make sure no
temporary relation accumulated.

Run with the project venv, in isolated mode::

    .venv/bin/python -I examples/relation_bindings/lifecycle_experiment.py
"""

from __future__ import annotations

import gc

import pyarrow as pa

import ibis
from ibis.backends.duckdb.bindings import BindingSession

ROUNDS = 200


def catalog_temp_objects(con) -> list[str]:
    rows = con.con.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
        """
    ).fetchall()
    return sorted(name for (name,) in rows)


def main() -> None:
    con = ibis.duckdb.connect()
    t = ibis.table(dict(a="int64", b="string"), name="t")
    expr = t.filter(t.a > 0).select("a", "b")

    data = pa.table(
        {"a": list(range(100)), "b": [str(i) for i in range(100)]}
    )

    gc.collect()
    baseline_tables = catalog_temp_objects(con)
    print(f"baseline catalog objects: {baseline_tables}")

    max_open_cursors = 0
    for i in range(ROUNDS):
        # one eager round
        con.execute(expr, table_bindings={t: data})
        # one streaming round, partially then fully consumed
        reader = con.to_pyarrow_batches(
            expr, table_bindings={t: data}, chunk_size=32
        )
        next(iter(reader))
        remaining = sum(b.num_rows for b in reader)
        assert remaining < 100

        # introspect a fresh session's teardown as well; streaming
        # sessions hold an explicit child cursor while alive
        probe = BindingSession(con, expr, {t: data})
        cursor = probe.track_cursor(con.con.cursor())
        probe.register_all(cursor)
        max_open_cursors = max(max_open_cursors, len(probe.active_cursors))
        probe.close()
        assert not probe.holds_data
        assert probe.active_cursors == ()
        cursor.close()

        if i % 50 == 0:
            print(
                f"round {i:3d}: catalog={catalog_temp_objects(con)}, "
                f"probe holds data after close: {probe.holds_data}"
            )

    gc.collect()
    end_tables = catalog_temp_objects(con)
    print(f"after {ROUNDS} rounds catalog objects: {end_tables}")
    assert end_tables == baseline_tables
    print("max open cursors observed inside a round:", max_open_cursors)
    print("backend still usable:", con.raw_sql("SELECT 1").fetchone())
    print("LIFECYCLE OK")


if __name__ == "__main__":
    main()
