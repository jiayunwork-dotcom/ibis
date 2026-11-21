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
try:
    r = cur.execute("SELECT boom(a) AS x FROM b").fetch_record_batch(rows_per_batch=2)
    for batch in r:
        pass
except Exception as e:
    print("error stage ok:", type(e).__name__)
cur.unregister("b")
cur.close()
print("root ok:", con.execute("select 1").fetchone())
