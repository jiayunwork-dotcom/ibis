import duckdb
import pyarrow as pa

con = duckdb.connect()
d1 = pa.table({"a": range(100_000)})
d2 = pa.table({"a": range(100_000, 200_000)})
d3 = pa.table({"a": range(1_000_000, 1_001_000)})
name = "ibis_duckdb_binding_abc"

c1 = con.cursor(); c2 = con.cursor()
c1.register(name, d1); c2.register(name, d2)
r1 = c1.execute(f"SELECT a FROM {name}").fetch_record_batch(rows_per_batch=1024)
r2 = c2.execute(f"SELECT a FROM {name}").fetch_record_batch(rows_per_batch=1024)
b1 = r1.read_next_batch(); b2 = r2.read_next_batch()
print("first:", b1.column(0)[0].as_py(), b2.column(0)[0].as_py())

con.register(name, d3)
print("eager root:", con.execute(f"SELECT count(*) FROM {name}").fetchone())
con.unregister(name)

b1b = r1.read_next_batch(); b2b = r2.read_next_batch()
print("second starts:", b1b.column(0)[0].as_py(), b2b.column(0)[0].as_py())
r1.close()
pulled = b2.num_rows + b2b.num_rows
total = pulled + sum(b.num_rows for b in r2)
print("r2 total:", total)
c1.close(); c2.close()
