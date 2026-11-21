import duckdb
import pyarrow as pa

con = duckdb.connect()
d1 = pa.table({"a": range(100_000)})
d2 = pa.table({"a": range(100_000, 200_000)})
d3 = pa.table({"a": range(1_000_000, 1_001_000)})

c1 = con.cursor(); c2 = con.cursor()
c1.register("r1", d1); c2.register("r2", d2)
r1 = c1.execute("SELECT a FROM r1").fetch_record_batch(rows_per_batch=1024)
r2 = c2.execute("SELECT a FROM r2").fetch_record_batch(rows_per_batch=1024)
b1 = r1.read_next_batch(); b2 = r2.read_next_batch()
print("first:", b1.column(0)[0].as_py(), b2.column(0)[0].as_py())

# eager query on ROOT con (registering another replacement scan on root)
con.register("r3", d3)
print("eager root:", con.execute("SELECT count(*) FROM r3").fetchone())
con.unregister("r3")

b2b = r2.read_next_batch()
print("r2 second batch starts:", b2b.column(0)[0].as_py(), "rows:", b2b.num_rows)
r1.close()
total = b2.num_rows + b2b.num_rows + sum(b.num_rows for b in r2)
print("r2 total:", total)
c1.close(); c2.close()
