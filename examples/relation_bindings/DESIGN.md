# Per-call relation bindings for the DuckDB backend

## Problem

An ETL expression is authored once against `ibis.table(...)` placeholders
and must be executed repeatedly, once per batch of `pyarrow.Table` data,
without rebuilding the graph, without asking users to create/drop tables,
and without colliding with same-named tables that already live in the
DuckDB catalog. Streaming readers must support same-thread interleaving
and release every resource deterministically.

## Two organizations considered

### Option 1 — compiler context

Thread a `{UnboundTable node: relation name}` mapping through the compiler
and make `visit_UnboundTable` consult it; register the arrow data under
those names right before execution.

* *Pros*: no new operation node, no graph duplication.
* *Cons*: the compiler's translation entry point gains a mutable
  side-channel that every rule must remain consistent with; compile-only
  calls and execution calls would have to reconstruct the *same* name
  assignment independently (names cannot be random per call if the two
  paths must agree); shared subgraphs and view self-joins need explicit
  bookkeeping inside the context; registration lifecycle is decoupled from
  the graph objects it names.

### Option 2 — pure graph rewrite + session-owned resources (implemented)

Validate the mapping up front and produce a **new, non-mutated** graph in
which every bound `UnboundTable` is replaced by a backend-local
`BoundRelation` physical-table node carrying a generated relation name. A
short-lived `BindingSession` owns the DuckDB replacement scans (and, for
streaming, the child cursor). The standard compilation pipeline then sees
ordinary physical tables — CTE extraction, join/window lowering and view
handling need no special cases.

* *Pros*:
  * compile/execute necessarily agree: both run the same pure function of
    the expression and the mapping; relation names are derived
    deterministically from the placeholder node's structural hash, so
    within one process two compilations of one mapping emit
    byte-identical SQL (Python's hash seed randomizes the numeric hash
    across processes, which only changes the generated name -- identity
    semantics are unaffected);
  * identity semantics (`node == node`, never table-name identity) fall
    out of Ibis's existing structural node equality, so same-named,
    different-schema placeholders get distinct relations and a shared
    subgraph / self-joined view shares one relation automatically;
  * the caller's expression and mapping are never mutated;
  * validation completes before a single catalog call, so a rejected call
    cannot leave partial state.
* *Costs*: a copy of the graph's node spine (linear in node count; shared
  subgraphs remain shared) and one small backend-local op class.

## How a binding travels through the system

1. **Backend selection & graph** (`ibis/expr/types/core.py`). The four
   entry points (`execute`, `compile`, `to_pyarrow`,
   `to_pyarrow_batches`) accept `table_bindings=` and an explicit
   `backend=`. With bindings present, an otherwise-unbound expression is
   allowed to resolve to the default backend; without them the historical
   resolution rules and error are unchanged.
2. **Validation & rewrite** (`ibis/backends/duckdb/bindings.py`).
   `BindingSession` normalizes expression keys to `UnboundTable` ops,
   rejects extra/foreign/already-bound keys and non-arrow values, requires
   every unbound table in the graph to be covered, and checks each arrow
   table for exact column-set, exact supported arrow type, column order
   (reordering is applied to a rebuilt table, never to the caller's data),
   non-nullable violations and empty-but-structured input. It then rewrites
   the graph to `BoundRelation` nodes.
3. **SQL compilation** (`ibis/backends/sql/compilers/duckdb.py`).
   `visit_BoundRelation` emits an ordinary quoted table reference; nothing
   else in the compiler changes. `Backend.compile` performs validation and
   rewrite only — it never registers data, reads rows, mutates the catalog
   or expands rows into `VALUES`.
4. **Registration & result conversion**
   (`ibis/backends/duckdb/__init__.py`). Eager calls register on the
   backend connection and materialize inside the binding context so
   teardown only happens after the data is pulled. Streaming calls create
   one child connection (`con.cursor()`) per reader, register there, and
   return a `BindingReader` that owns the session.

## Resource ownership and lifecycle

* DuckDB replacement scans (`con.register(name, arrow_table)`) are
  connection local: a registration on a child cursor is invisible
  elsewhere, and shadowing a catalog table on one connection never touches
  the catalog table itself (verified in `verification.py`).
* **Eager**: one session per call, registered on the backend connection,
  closed in a `finally` after materialization. On failure, names already
  registered are rolled back before the exception propagates.
* **Streaming**: one child connection per reader keeps its own cursor and
  catalog entries, so two readers interleave independently and an eager
  query on the backend cannot clobber either. `BindingReader.close()`
  (also invoked on exhaustion or error) unregisters the names, closes the
  child cursor and drops the session's arrow references — deterministically,
  without relying on `gc.collect()`, and never closes the backend itself.
  No output is buffered in Python beyond the batch currently being read.

## Supported column types

`int64`, `float64`, `string`, `boolean`, and microsecond-precision,
timezone-naive `timestamp`. Arrow types must match the declaration
exactly; no implicit conversion is performed. Nullability differences are
accepted except that a non-nullable declaration rejects nulls in the
incoming column.

## Experiments

From the repository root with the project virtualenv:

```bash
.venv/bin/python -I examples/relation_bindings/usage_example.py
.venv/bin/python -I examples/relation_bindings/verification.py
.venv/bin/python -I examples/relation_bindings/lifecycle_experiment.py
.venv/bin/python -I examples/relation_bindings/streaming_memory_experiment.py
```

The lifecycle script runs 200 mixed rounds and asserts the catalog, cursor
and data-reference state returns to baseline. The streaming script builds
a >1M-row input before measurement and separately reports input
footprint, DuckDB compute residency and output buffering, comparing
time-to-first-batch against full eager materialization.

## Measured results (this environment)

Python 3.12.16, DuckDB 1.1.3, PyArrow 17.0.0, Linux aarch64.

`lifecycle_experiment.py` (200 rounds, each round = one eager call + one
streaming call + one explicit session probe):

```
baseline catalog objects: []
after 200 rounds catalog objects: []
max open cursors observed inside a round: 1
backend still usable: (1,)
LIFECYCLE OK
```

`streaming_memory_experiment.py` (1,000,000 rows, unsorted projection):

```
input rows: 1,000,000
input arrow size: 29.9 MB
RSS delta attributable to input: +44.8 MB      # input footprint

streaming (to_pyarrow_batches):
  time to first batch (1024 rows): 13.3 ms     # no full materialization
  total time:                        29.3 ms
  batches:                           977
  largest single batch:              31.1 KB   # new output buffering per batch
  RSS at first batch vs before call: +36.1 MB  # DuckDB compute residency
eager (to_pyarrow):
  time to full table: 21.5 ms
  output arrow size:  30.3 MB
  RSS delta vs before: +61.3 MB                # includes whole output
```

Interpretation: the input itself accounts for ~45 MB of RSS (caller data
that exists before any binding call). DuckDB's streaming execution holds
roughly the same working set while producing 1024-row batches whose largest
Python-side footprint is ~31 KB, so the reader never materializes the full
~30 MB output before returning. The eager path, by construction, adds the
complete output table on top of the working set and pays the full
materialization time before returning.

## How to rerun

```bash
# create the environment used here (uv will fetch CPython 3.12)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
    "duckdb==1.1.3" "pyarrow==17.0.0" pyarrow_hotfix packaging rich \
    pytest hypothesis pytest-benchmark -e .

# new feature tests
.venv/bin/python -I -m pytest ibis/backends/duckdb/tests/test_relation_bindings.py -q

# affected existing suites (the deselected failures are environment/data
# issues that also fail on the unmodified tree, verified via git stash)
.venv/bin/python -I -m pytest \
    ibis/backends/duckdb/tests/test_client.py \
    ibis/backends/duckdb/tests/test_register.py \
    ibis/backends/duckdb/tests/test_catalog.py \
    ibis/backends/duckdb/tests/test_udf.py \
    ibis/backends/sql/tests/ ibis/tests/expr -q

# standalone examples and experiments; the .out files next to the scripts
# hold the output captured during the run documented above
.venv/bin/python -I examples/relation_bindings/usage_example.py
.venv/bin/python -I examples/relation_bindings/verification.py
.venv/bin/python -I examples/relation_bindings/lifecycle_experiment.py
.venv/bin/python -I examples/relation_bindings/streaming_memory_experiment.py
```
