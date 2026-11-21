import duckdb
import pyarrow as pa

con = duckdb.connect()
d1 = pa.table({"a": range(100000)})
d2 = pa.table({"a": range(100000, 200000)})
con.register("r1", d1)
con.register("r2", d2)

cur1 = con.execute("SELECT a FROM r1")
print("type of execute result:", type(cur1), cur1 is con)
cur2 = con.execute("SELECT a FROM r2")
print("cur1 is cur2:", cur1 is cur2)
# try explicit cursors
c1 = con.cursor()
c2 = con.cursor()
cc1 = c1.execute("SELECT a FROM r1")
cc2 = c2.execute("SELECT a FROM r2")
print("cc1 is c1:", cc1 is c1)
b1 = next(cc1.fetch_record_batch(rows_per_batch=1024))
b2 = next(cc2.fetch_record_batch(rows_per_batch=1024))
print("first rows:", b1.column(0)[0].as_py(), b2.column(0)[0].as_py())
print("eager on root con mid-stream:", con.execute("SELECT count(*) FROM r1").fetchone())
# eager using a fresh cursor
ec = con.cursor()
print("eager via fresh cursor:", ec.execute("SELECT count(*) FROM r2").fetchone())
ec.close()
n1 = 1; n2 = 1
for b in cc1.fetch_record_batch(rows_per_batch=4096): n1 += 1
for b in cc2.fetch_record_batch(rows_per_batch=4096): n2 += 1
print("batch counts:", n1, n2)
# close behavior
cc3 = con.cursor().execute("SELECT a FROM r1")
gen = cc3.fetch_record_batch(rows_per_batch=1024)
b = next(gen)
print("got batch rows:", b.num_rows)
cc3.close()
try:
    next(gen)
except Exception as e:
    print("after close, next(gen):", type(e).__name__, e)
# are registered views still usable after cursor close?
print("table still registered:", con.execute("SELECT count(*) FROM r1").fetchone())
c1.close(); c2.close()
con.unregister("r1"); con.unregister("r2")
print("done; tables:", con.execute("SHOW TABLES").fetchall())
