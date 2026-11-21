"""Streaming and memory experiment on >= 1,000,000 bound rows.

The expression is a deliberately simple, unsorted row-wise projection so
that no sort or aggregation materializes the whole input in DuckDB. The
input ``pyarrow.Table`` is created *before* any measurement, and three
quantities are kept separate:

1. **input footprint** -- the resident-memory delta of building the
   pyarrow table (the caller's data, not binding machinery);
2. **DuckDB compute footprint** -- resident memory held while executing the
   projection, measured as the process RSS after the query starts
   producing output;
3. **new output buffering** -- extra Python-side residency while reading
   batches one at a time and discarding every batch immediately.

The streaming path must return its first batch without materializing the
full result, and the eager path is expected to pull everything up front;
the experiment reports both timings and RSS deltas so the difference is
visible. Run in isolated mode::

    .venv/bin/python -I examples/relation_bindings/streaming_memory_experiment.py
"""

from __future__ import annotations

import gc
import time

import pyarrow as pa

import ibis

ROWS = 1_000_000
CHUNK = 1024


def rss_mb() -> float:
    # current resident set size in MB (VmRSS can fall after teardown,
    # unlike ru_maxrss which is a high-water mark)
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                kb = int(line.split()[1])
                return kb / 1024.0
    raise RuntimeError("VmRSS not available")


def main() -> None:
    con = ibis.duckdb.connect()
    t = ibis.table(dict(a="int64", b="float64", c="string"), name="t")
    # unsorted row-wise projection: arithmetic + string prefix, no shuffle
    expr = t.mutate(
        a2=t.a + 1,
        b2=t.b * 2.0,
    ).select("a2", "b2", "c")

    import numpy as np

    gc.collect()
    rss_before_input = rss_mb()
    # input is created before any timed/measured binding work
    data = pa.table(
        {
            "a": pa.array(range(ROWS), type=pa.int64()),
            "b": pa.array(np.random.random(ROWS), type=pa.float64()),
            "c": pa.array([f"row-{i}" for i in range(ROWS)]),
        }
    )
    input_bytes = data.nbytes
    print(f"input rows: {ROWS:,}")
    print(f"input arrow size: {input_bytes / 1e6:.1f} MB")

    gc.collect()
    rss_after_input = rss_mb()
    print(
        f"RSS delta attributable to input: "
        f"{rss_after_input - rss_before_input:+.1f} MB"
    )

    # ------------------------------------------------------------------
    # streaming path: time to first batch vs total, and output buffering
    # ------------------------------------------------------------------
    gc.collect()
    rss_before_stream = rss_mb()
    t0 = time.perf_counter()
    reader = con.to_pyarrow_batches(
        expr, table_bindings={t: data}, chunk_size=CHUNK
    )
    first = reader.read_next_batch()
    t_first = time.perf_counter() - t0
    rss_first_batch = rss_mb()
    held_after_first = rss_first_batch - rss_before_stream
    assert first.num_rows == CHUNK

    rows_seen = first.num_rows
    batches_seen = 1
    max_single_batch = first.nbytes
    t0_rest = time.perf_counter()
    for batch in reader:
        rows_seen += batch.num_rows
        batches_seen += 1
        max_single_batch = max(max_single_batch, batch.nbytes)
        # batches are dropped immediately: nothing accumulates in Python
    t_total = (t_first) + (time.perf_counter() - t0_rest)
    rss_after_stream = rss_mb()

    assert rows_seen == ROWS, rows_seen
    print(
        "\nstreaming (to_pyarrow_batches):\n"
        f"  time to first batch ({CHUNK} rows): {t_first * 1000:.1f} ms\n"
        f"  total time:                        {t_total * 1000:.1f} ms\n"
        f"  batches:                           {batches_seen:,}\n"
        f"  largest single batch:              "
        f"{max_single_batch / 1e3:.1f} KB\n"
        f"  RSS at first batch vs before call: "
        f"{held_after_first:+.1f} MB (duckdb compute + input)\n"
        f"  RSS after full drain:              "
        f"{rss_after_stream - rss_before_stream:+.1f} MB"
    )
    # the reader never held the whole result in Python: the biggest batch
    # is orders of magnitude smaller than the full output
    assert max_single_batch < input_bytes / 50

    # ------------------------------------------------------------------
    # eager path: full materialization for comparison
    # ------------------------------------------------------------------
    gc.collect()
    rss_before_eager = rss_mb()
    t0 = time.perf_counter()
    eager = con.to_pyarrow(expr, table_bindings={t: data})
    t_eager = time.perf_counter() - t0
    rss_after_eager = rss_mb()
    assert eager.num_rows == ROWS
    print(
        "\neager (to_pyarrow):\n"
        f"  time to full table: {t_eager * 1000:.1f} ms\n"
        f"  output arrow size:  {eager.nbytes / 1e6:.1f} MB\n"
        f"  RSS delta vs before: "
        f"{rss_after_eager - rss_before_eager:+.1f} MB"
    )

    assert con.list_tables() == []
    print(
        f"\nstreaming first-batch wait was {t_eager / t_first:.1f}x faster "
        "than full eager materialization"
    )
    print("STREAMING/MEMORY OK")


if __name__ == "__main__":
    main()
