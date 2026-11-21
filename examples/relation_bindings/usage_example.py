"""Minimal usage example for per-call relation bindings.

An ETL expression is built once from abstract ``ibis.table`` placeholders
and then fed different pyarrow batches at execution time, without creating
or dropping any table in the DuckDB catalog::

    .venv/bin/python -I examples/relation_bindings/usage_example.py
"""

from __future__ import annotations

import pyarrow as pa

import ibis

con = ibis.duckdb.connect()

# 1. define the reusable expression against abstract placeholders
orders = ibis.table(
    dict(order_id="int64", customer="string", amount="float64"),
    name="orders",
)
pipeline = (
    orders.filter(orders.amount > 0)
    .group_by("customer")
    .aggregate(
        order_count=orders.order_id.count(),
        revenue=orders.amount.sum(),
    )
    .order_by("customer")
)

# 2. run it repeatedly against different batches
for day, rows in [
    (
        "2026-11-20",
        pa.table(
            {
                "order_id": [1, 2, 3, 4],
                "customer": ["ada", "ada", "grace", "grace"],
                "amount": [10.0, 5.0, 7.5, -3.0],
            }
        ),
    ),
    (
        "2026-11-21",
        pa.table(
            {
                "order_id": [5, 6],
                "customer": ["ada", "lin"],
                "amount": [20.0, 8.0],
            }
        ),
    ),
]:
    result = con.execute(pipeline, table_bindings={orders: rows})
    print(f"--- batch {day} ---")
    print(result)

# 3. all four entry points accept the same mapping
batch = pa.table(
    {
        "order_id": [7],
        "customer": ["lin"],
        "amount": [4.0],
    }
)
print("execute:\n", con.execute(pipeline, table_bindings={orders: batch}))
print("to_pyarrow:\n", con.to_pyarrow(pipeline, table_bindings={orders: batch}))
reader = con.to_pyarrow_batches(pipeline, table_bindings={orders: batch})
print("batches rows:", sum(b.num_rows for b in reader))
print("compile:\n", con.compile(pipeline, table_bindings={orders: batch}))

# 4. the same keyword works on the expression itself, backend optional
print(
    "expression entry point:\n",
    pipeline.execute(backend=con, table_bindings={orders: batch}),
)

# 5. scalar params and relation bindings compose
threshold = ibis.param("float64")
parametric = orders.filter(orders.amount > threshold)
print(
    "params + bindings:\n",
    con.execute(
        parametric,
        params={threshold: 6.0},
        table_bindings={orders: batch},
    ),
)

# nothing was ever created in the catalog
assert con.list_tables() == []
print("\ncatalog stays empty:", con.list_tables())
