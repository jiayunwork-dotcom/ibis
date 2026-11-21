import duckdb
import pyarrow as pa

con = duckdb.connect()

# 1. existing persistent table with same name as a replacement scan
con.execute("CREATE TABLE t (a INTEGER)")
con.execute("INSERT INTO t VALUES (42)")
data = pa.table({"a": [1, 2, 3]})
con.register("t", data)
print("after register, select:", con.execute("SELECT sum(a) FROM t").fetchall())
con.unregister("t")
print("after unregister, persistent table still there:", con.execute("SELECT * FROM t").fetchall())

# 2. two concurrent streaming cursors interleaved + eager query in between
d1 = pa.table({"a": range(100000)})
d2 = pa.table({"a": range(100000, 200000)})
con.register("r1", d1)
con.register("r2", d2)
cur1 = con.execute("SELECT a FROM r1")
cur2 = con.execute("SELECT a FROM r2")
b1 = next(cur1.fetch_record_batch(rows_per_batch=1024))
b2 = next(cur2.fetch_record_batch(rows_per_batch=1024))
print("first rows:", b1.column(0)[0].as_py(), b2.column(0)[0].as_py())
# eager query on same con
print("eager mid-stream:", con.execute("SELECT count(*) FROM r1").fetchone())
b1b = next(cur1.fetch_record_batch(rows_per_batch=1024))
b2b = next(cur2.fetch_record_batch(rows_per_batch=1024))
print("second batch first rows:", b1b.column(0)[0].as_py(), b2b.column(0)[0].as_py())
# fully drain r1
n = 1
for _ in cur1.fetch_record_batch(rows_per_batch=4096):
    n += 1
print("r1 batch count:", n)
# cur2 still works
total = sum(b.num_rows for b in cur2.fetch_record_batch(rows_per_batch=4096)) + b2.num_rows
print("r2 total rows incl first batch:", total)
con.unregister("r1"); con.unregister("r2")

# 3. error mid-query then reuse
con.register("r3", pa.table({"a": [1, 0, 3]}))
try:
    con.execute("SELECT 100/a FROM r3").fetchall()
except Exception as e:
    print("division error:", type(e).__name__)
print("reuse after error:", con.execute("SELECT count(*) FROM r3").fetchone())
con.unregister("r3")
print("tables now:", con.execute("SHOW TABLES").fetchall())
