import duckdb
import pyarrow as pa

con = duckdb.connect()
d1 = pa.table({"a": range(100000)})
d2 = pa.table({"a": range(100000, 200000)})
d3 = pa.table({"a": range(1000000, 1001000)})

# register each name on its own cursor
c1 = con.cursor(); c2 = con.cursor(); ce = con.cursor()
c1.register("r1", d1)
c2.register("r2", d2)
print("r1 on c2?")
try:
    c2.execute("SELECT count(*) FROM r1").fetchone()
except Exception as e:
    print("  no:", type(e).__name__)
cc1 = c1.execute("SELECT a FROM r1")
cc2 = c2.execute("SELECT a FROM r2")
b1 = next(cc1.fetch_record_batch(rows_per_batch=1024))
b2 = next(cc2.fetch_record_batch(rows_per_batch=1024))
print("first rows:", b1.column(0)[0].as_py(), b2.column(0)[0].as_py())

# third eager binding on same backend
ce.register("r3", d3)
print("eager third binding mid-stream:", ce.execute("SELECT count(*) FROM r3").fetchone())
ce.unregister("r3")
ce.close()

# interleaving continued
b1b = next(cc1.fetch_record_batch(rows_per_batch=4096))
b2b = next(cc2.fetch_record_batch(rows_per_batch=4096))
print("next batch starts:", b1b.column(0)[0].as_py(), b2b.column(0)[0].as_py())

# drain c2 fully
total2 = b2.num_rows + sum(b.num_rows for b in cc2.fetch_record_batch(rows_per_batch=4096))
print("r2 total:", total2)
# drain c1
total1 = b1.num_rows + b1b.num_rows + sum(b.num_rows for b in cc1.fetch_record_batch(rows_per_batch=4096))
print("r1 total:", total1)
print("r1 still registered on c1 after drain?")
try:
    print(c1.execute("SELECT count(*) FROM r1").fetchone())
except Exception as e:
    print("  ", type(e).__name__, e)
c1.close(); c2.close()
# root con unaffected
print("root con still alive:", con.execute("SELECT 1").fetchone())
