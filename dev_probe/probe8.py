import duckdb
import pyarrow as pa
con = duckdb.connect()

def boom(x):
    if x > 5:
        raise RuntimeError("boom from udf")
    return x

cur = con.cursor()
cur.create_function("boom", boom, ["BIGINT"], "BIGINT")
cur.register("b", pa.table({"a": range(20)}))
r = cur.execute("SELECT boom(a) AS x FROM b").fetch_record_batch(rows_per_batch=2)
try:
    for batch in r:
        print(batch.to_pydict())
except Exception as e:
    print("fetch error:", type(e).__name__, e)
# now cleanup explicitly; unregister + close should not raise
cur.unregister("b")
cur.close()
print("root ok:", con.execute("select 1").fetchone())

# how many cursors do we accumulate? check describe
c2 = con.cursor()
c2.register("x", pa.table({"a":[1]}))
r2 = c2.execute("select * from x").fetch_record_batch(rows_per_batch=1)
next(iter(r2))
print("registered views on child (temp):", c2.execute("select table_name from information_schema.tables where table_schema='main'").fetchall())
r2.close(); c2.unregister("x"); c2.close()
print("done")
