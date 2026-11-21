"""Manual lifecycle probe for relation bindings (not part of pytest suite)."""

import gc

import ibis
import pyarrow as pa
from ibis import udf

con = ibis.duckdb.connect()
t = ibis.table(dict(a='int64'), name='t')
expr = t

d1 = pa.table({'a': range(100_000)})
d2 = pa.table({'a': range(100_000, 200_000)})
d3 = pa.table({'a': range(1_000_000, 1_001_000)})

r1 = con.to_pyarrow_batches(expr, table_bindings={t: d1}, chunk_size=1024)
r2 = con.to_pyarrow_batches(expr, table_bindings={t: d2}, chunk_size=1024)

b1 = r1.read_next_batch()
b2 = r2.read_next_batch()
assert b1.column(0)[0].as_py() == 0
assert b2.column(0)[0].as_py() == 100_000

# third eager binding interleaved on the SAME backend
eager = con.execute(expr, table_bindings={t: d3})
assert len(eager) == 1000
print("eager third binding interleaved ok")

b1b = r1.read_next_batch()
b2b = r2.read_next_batch()
assert b1b.column(0)[0].as_py() == 1024
assert b2b.column(0)[0].as_py() == 101_024
print("interleaved cursors independent ok")

r1.close()
total = b2.num_rows + b2b.num_rows + sum(b.num_rows for b in r2)
assert total == 100_000, total
print("early close + full drain ok, total =", total)
assert con.list_tables() == []
print("catalog clean after streaming")


# error path: python udf that raises mid-query
@udf.scalar.python
def boom(x: int) -> int:
    if x > 5:
        raise RuntimeError("boom from udf")
    return x


err_expr = t.mutate(x=boom(t.a))
try:
    reader = con.to_pyarrow_batches(err_expr, table_bindings={t: d1}, chunk_size=2)
    reader.read_all()
except Exception as e:
    print("batch/execution error surfaced:", type(e).__name__)
else:
    raise AssertionError("expected an error")

# session must have cleaned up already
assert con.list_tables() == []
print("cleanup after fetch error ok")

# subsequent correct call still works
out = con.execute(expr, table_bindings={t: d3})
assert len(out) == 1000
print("reuse after error ok:", len(out))

# gc does not resurrect anything
gc.collect()
assert con.list_tables() == []
print("all probes passed")
