import duckdb
import pyarrow as pa

con = duckdb.connect()
con.execute("CREATE TABLE user_t (a INTEGER, b VARCHAR)")
con.execute("INSERT INTO user_t VALUES (42, 'persisted')")
con.execute("CREATE VIEW user_v AS SELECT a*2 AS a FROM user_t")

# child cursor sees persistent catalog objects
c = con.cursor()
print("child sees table:", c.execute("SELECT * FROM user_t").fetchall())
print("child sees view:", c.execute("SELECT * FROM user_v").fetchall())

# replacement scan registered on child shadows on child only
data = pa.table({"a": [1,2,3]})
c.register("user_t", data)
print("child shadowed:", c.execute("SELECT sum(a) FROM user_t").fetchone())
print("root untouched:", con.execute("SELECT * FROM user_t").fetchall())
c.unregister("user_t")
print("child sees original again:", c.execute("SELECT * FROM user_t").fetchall())

# errors: division by zero mid fetch
c.register("bad", pa.table({"a":[1,0,3]}))
cur = c.cursor()
try:
    cur.execute("SELECT 10/a FROM bad").fetchall()
except Exception as e:
    print("query error:", type(e).__name__)
# cursor usable again?
print("cursor reuse:", cur.execute("SELECT 1").fetchone())
# and root usable
print("root usable:", con.execute("SELECT 1").fetchone())
c.unregister("bad")
c.close()
print("final tables:", con.execute("SHOW TABLES").fetchall())
print("final views:", con.execute("SELECT view_name FROM duckdb_views WHERE NOT temporary").fetchall())
