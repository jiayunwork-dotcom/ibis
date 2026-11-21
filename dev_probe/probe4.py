import duckdb
import pyarrow as pa

con = duckdb.connect()
d1 = pa.table({"a": range(100000)})
d2 = pa.table({"a": range(100000, 200000)})
d3 = pa.table({"a": range(1000000, 1001000)})

c1 = con.cursor(); c2 = con.cursor()
c1.register("r1", d1)
c2.register("r2", d2)
gen1 = c1.execute("SELECT a FROM r1").fetch_record_batch(rows_per_batch=1024)
gen2 = c2.execute("SELECT a FROM r2").fetch_record_batch(rows_per_batch=1024)
b1 = next(gen1); b2 = next(gen2)
print("first rows:", b1.column(0)[0].as_py(), b2.column(0)[0].as_py())

# third eager binding on the same backend
ce = con.cursor()
ce.register("r3", d3)
print("eager third binding mid-stream:", ce.execute("SELECT count(*) FROM r3").fetchone())
ce.unregister("r3"); ce.close()

# interleave kept generators
b1b = next(gen1); b2b = next(gen2)
print("next batch starts:", b1b.column(0)[0].as_py(), b2b.column(0)[0].as_py())

# drain c2 fully
n2 = b2.num_rows + b2b.num_rows + sum(b.num_rows for b in gen2)
print("r2 total:", n2)
n1 = b1.num_rows + b1b.num_rows + sum(b.num_rows for b in gen1)
print("r1 total:", n1)

# close one early on a fresh streaming query
c3 = con.cursor()
c3.register("r1b", d1)
gen3 = c3.execute("SELECT a FROM r1b").fetch_record_batch(rows_per_batch=1024)
b = next(gen3)
print("c3 first batch:", b.num_rows)
c3.close()
try:
    next(gen3)
except Exception as e:
    print("after close:", type(e).__name__)

c1.close(); c2.close()
print("root con still alive:", con.execute("SELECT 1").fetchone())
print("OK")
