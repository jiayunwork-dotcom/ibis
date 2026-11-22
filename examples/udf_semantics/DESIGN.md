# Scalar UDF configuration and evaluation semantics (DuckDB)

## Problem

`@ibis.udf.scalar.python` and `@ibis.udf.scalar.pyarrow` accept
`null_handling`, `exception_handling` and `side_effects`, but none of those
options reached `duckdb.DuckDBPyConnection.create_function`. Three
symptoms followed:

1. `null_handling="special"` was ignored, so a UDF that wanted to handle
   `NULL` never saw it, and `ibis.null().cast("int64")` evaluated to `None`
   instead of the function's result.
2. `exception_handling="return_null"` was ignored, so a `ValueError`
   propagated out of the query instead of becoming `NULL`.
3. `side_effects=True` was ignored by both Ibis and DuckDB. Ibis compiled
   the UDF like a pure value, so a duplicated call site collapsed into one
   evaluation and a named result was inlined back into every reference.

## Two organizations considered

### Option 1 — thread an "impure" flag through every rewrite

Teach each relational rewrite (project merging, filter/project reordering,
CTE extraction, deduplication) to consult a per-expression set of
side-effecting call sites and bail out when one is present.

* *Pros*: local reasoning per rewrite.
* *Cons*: the same condition would have to be repeated in every rule (SQL
  `sqlize`, expression `simplify`, complexity scoring, dereferencing); a
  missed rule silently reintroduces the bug; and it would still collapse
  *different* call sites of the same function into one node unless the
  node's identity distinguished them.

### Option 2 — mark the node impure and let existing rules preserve it (implemented)

When a UDF is declared `side_effects=True`, its generated operation class
also inherits `ops.Impure`. Ibis already has an evaluation-boundary concept
for operators such as `RandomScalar`/`RandomUUID`; reusing it means every
existing rule that guards on `Impure` (dereferencing backtracking, select
merging, complexity scoring) preserves the call boundary, while pure UDFs
keep the exact optimization behaviour they had before.

* *Pros*:
  * one declaration, honoured by all existing rewrites;
  * a pure UDF is still a plain `Value`, so inlining and deduplication keep
    working (regression-tested);
  * the door is not closed for the compiler to deliberately duplicate a
    pure call.
* *Costs*: the rewrite rules still need to make the *named-result* case
  stable — an impure value must not be inlined into a later projection
  because that turns one evaluation into many. This is exactly what the
  `Impure` guards in `ibis/expr/rewrites.py` do (see below). The final
  plan keeps the named value in a subquery and references the column by
  name, so predicate pushdown cannot re-evaluate it, duplicate references
  read one evaluation, and streaming continues to work.

The alternative — disabling graph optimization globally or treating every
UDF as impure — was rejected: it would pessimize all deterministic UDFs
and violate the "existing pure results and optimizations must be
preserved" requirement.

## What changed

1. **Node identity and impurity** (`ibis/expr/operations/udf.py`). The
   node class is generated with `(base, ops.Impure)` when
   `kwargs.get("side_effects")` is true. `__config__` still stores the raw
   decorator keyword arguments, so the compiler and backend registration
   read the same dictionary.

2. **Registration** (`ibis/backends/duckdb/__init__.py`). `_register_udf`
   forwards `**udf_node.__config__` to `con.create_function`, so
   `null_handling`, `exception_handling` and `side_effects` become DuckDB
   function properties. Options that were not supplied are absent, so
   DuckDB's defaults (intercept `NULL`, propagate exceptions, treat as
   pure) are unchanged. `_register_python_udf` and `_register_pyarrow_udf`
   share this path, so Arrow UDFs receive the same configuration.

3. **Expression rewrites** (`ibis/expr/rewrites.py`). `subsequent_projects`
   and `reorder_filter_project` now return early when the relation they
   would inline carries an impure value. Without this, dereferencing a
   named field would substitute the UDF expression back into the consumer
   and evaluate it again.

4. **SQL rewrites** (`ibis/backends/sql/rewrites.py`). The complexity
   score treats `Impure` nodes as maximally complex, so a
   complexity-limits-based merge never favors inlining a side-effecting
   value. The existing select-merge rule already lists `ops.Impure` among
   its blocking nodes, which is what produces the
   `SELECT ... FROM (SELECT udf(...) AS z ...) WHERE z ...` shape.

## How the evaluation boundary is preserved

* **Two independent call sites** (`select(a=f(x), b=f(x))`): both call
  expressions are distinct nodes; the projection compiles both, so the
  UDF runs twice per row. DuckDB itself is told `side_effects=True`, so
  it does not common-subexpression-eliminate them either. Result: six
  evaluations for three rows.
* **One named result, two references** (`m = t.mutate(z=f(x));
  m.select(a=m.z, b=m.z+1)`): `z` is materialized once in a subquery; the
  outer projection reads the named column. Result: three evaluations and
  `b == a + 1` on every row.
* **Explicitly named, then renamed/filtered/re-projected**: the same
  subquery boundary is retained; the named column is read rather than the
  UDF expression being re-emitted.
* **Compile only**: `con.compile()` never executes the callback; it emits
  SQL. Verified with a counter.
* **Streaming**: `to_pyarrow_batches` registers the UDF and returns a
  reader; batches are produced by DuckDB's cursor. A bounded prefix may be
  evaluated before the first batch is returned, but the whole result is
  never buffered in Python. Python callbacks stay on the database side.

## Arrow (`scalar.pyarrow`)

Arrow UDFs are registered with `type=duckdb.functional.ARROW` plus the same
forwarded configuration. The callback receives a `ChunkedArray` per batch,
so `null_handling="special"` must be used to see `NULL` rows, and
`exception_handling="return_null"` turns a raised exception into `NULL` for
that batch. Tests assert per-batch invocation (call count strictly less
than the row count), never one call per row.

## Differences from eager execution

`execute()` uses a `DuckDBPyRelation` (`con.sql`) path while
`to_pyarrow_batches` uses a cursor. Both share `_register_udfs`, so the
configuration is identical. `execute` returns pandas, which can render a
`NULL` integer column as `NaN`; the tests therefore assert null-handling
semantics through `to_pyarrow` (arrow-level `None`) or by fills, and use
`execute` for counts and relations.

## Experiments

From the repository root with the project virtualenv:

```bash
.venv/bin/python -I examples/udf_semantics/verification.py
```

It prints the direct-DuckDB reference, then the three fixed behaviours
through `execute`, `to_pyarrow` and `to_pyarrow_batches`, the Arrow batch
protocol, and a streaming-memory measurement. `verify.out` holds the
captured output.

## Measured results (this environment)

Python 3.12.16, DuckDB 1.1.3, PyArrow 17.0.0, Linux aarch64.

```
reference: two call sites -> 6 calls; one named site -> 3 calls
null_handling='special': scalar NULL -> 77; column mixed -> [77, 2, 3]
exception_handling: return_null -> None; default propagates -> InvalidInputException
side_effects=True: two call sites -> 6 calls; named reused -> 3 calls, b == a + 1
pyarrow UDF: 200000 rows in 98 per-batch calls (not per row)
streaming: 4096-row first batch; full payload would be 40 MB;
           largest single batch 0.84 MB; python peak 0.03 MB;
           200000 rows drained, no full-result buffering
```

The callback had already run for a bounded prefix before the first batch was
returned (DuckDB pipelines ahead), but neither the reader nor the test ever
holds the whole output: the test keeps only an integer counter and the batch
currently in flight.

## How to rerun

```bash
# create the environment used here (uv fetches CPython 3.12)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
    "duckdb==1.1.3" "pyarrow==17.0.0" -e ".[duckdb]" \
    pytest hypothesis pytest-benchmark pytest-randomly pytest-snapshot filelock mock

# new regression suite
NIX_SANDBOX=1 IN_NIX_SHELL=pur .venv/bin/python -m pytest \
    ibis/backends/duckdb/tests/test_udf_semantics.py -q -p no:randomly

# existing suites that exercise UDFs, rewrites, SQL compilation and bindings
NIX_SANDBOX=1 IN_NIX_SHELL=pur .venv/bin/python -m pytest \
    ibis/backends/duckdb/tests/test_udf.py \
    ibis/backends/duckdb/tests/test_relation_bindings.py \
    ibis/backends/sql/tests ibis/tests/expr ibis/expr/tests -q
```

The `NIX_SANDBOX`/`IN_NIX_SHELL` pair marks the run as sandboxed so the
DuckDB test fixture does not try to download the optional spatial/httpfs
extensions (that download is the only reason those fixtures fail in an
offline sandbox; it is unrelated to these changes and also fails on the
unmodified tree).
