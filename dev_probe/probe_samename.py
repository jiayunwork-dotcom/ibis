import duckdb
import pyarrow as pa

con = duckdb.connect()
d2 = pa.table({"a": range(100_000, 200_000)})
d3 = pa.table({"a": range(1_000_000, 1_001_000)})

c2 = con.cursor()
# same deterministic name on child and later on root
name = "ibis_duckdb_binding_abc"
c2.register(name, d2)
r2 = c2.execute(f"SELECT a FROM {name}").fetch_record_batch(rows_per_batch=1024)
b2 = r2.read_next_batch()
print("first starts:", b2.column(0)[0].as_py())

# eager session registers the SAME name on root
con.register(name, d3)
print("eager root:", con.execute(f"SELECT count(*) FROM {name}").fetchone())
con.unregister(name)

b2b = r2.read_next_batch()
print("second starts:", b2b.column(0)[0].as_py())
total = b2.num_rows + b2b.num_rows + sum(b.num_rows for b in r2)
print("r2 total:", total)
c2.close()
