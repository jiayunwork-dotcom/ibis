import duckdb
import pyarrow as pa

con = duckdb.connect()
empty = pa.table({"a": pa.array([], type=pa.int64()), "b": pa.array([], type=pa.string())})
cur = con.cursor()
cur.register("e", empty)
reader = cur.execute("SELECT a, b FROM e").fetch_record_batch(rows_per_batch=1024)
print("reader type:", type(reader).__name__)
batches = list(reader)
print("num batches on empty:", len(batches))
# explicit close without reading
cur2 = con.cursor()
cur2.register("e2", pa.table({"a": range(100000)}))
r2 = cur2.execute("SELECT a FROM e2").fetch_record_batch(rows_per_batch=1024)
b = next(iter(r2))
r2.close()
cur2.close()
print("closed fine")
# error during fetch
cur3 = con.cursor()
cur3.register("b3", pa.table({"a":[1,0,2]}))
r3 = cur3.execute("SELECT 10/a AS x FROM b3").fetch_record_batch(rows_per_batch=1)
try:
    list(r3)
except Exception as e:
    print("fetch error:", type(e).__name__)
cur3.close()
print("root ok:", con.execute("select 1").fetchone())
