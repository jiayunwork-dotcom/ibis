import duckdb
import pyarrow as pa
con = duckdb.connect()
cur3 = con.cursor()
cur3.register("b3", pa.table({"a":[1,0,2]}))
r3 = cur3.execute("SELECT 10/a AS x FROM b3").fetch_record_batch(rows_per_batch=1)
try:
    for b in r3:
        print(b.to_pydict())
except Exception as e:
    print("fetch error:", type(e).__name__, e)
# after error, reader state
try:
    next(iter(r3))
except Exception as e:
    print("re-read after error:", type(e).__name__)
cur3.close()
print("root ok:", con.execute("select 1").fetchone())
