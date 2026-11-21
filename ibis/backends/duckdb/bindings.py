"""Per-call relation bindings for the DuckDB backend.

An :class:`ibis.expr.operations.UnboundTable` created with :func:`ibis.table`
is only a structural placeholder: it carries a name and a schema but no data.
Executing an expression that contains one normally requires a table with that
name to exist in the connected database.

The machinery in this module takes a *binding map* -- from unbound table
*expressions* (identity follows Ibis node equality, i.e. structure, not the
table name) to concrete ``pyarrow.Table`` objects -- and makes a single
expression executable against an arbitrary batch of data, without touching
same-named user tables in the catalog.

Design overview
---------------
Two broad organizations can satisfy the required boundaries:

1. **Compile-time name context.** Thread a ``{node: relation_name}`` mapping
   through the compiler and teach ``visit_UnboundTable`` to consult it, then
   register the arrow data under those names right before execution.

2. **Pure graph rewrite (implemented here).** Validate the binding map up
   front and produce a *new* expression graph in which every bound
   ``UnboundTable`` node is replaced by a backend-local
   :class:`BoundRelation` node carrying a relation name. The rest of the
   compilation pipeline (CTE extraction, join/window lowering, view
   self-joins) sees an ordinary physical table and needs no special cases,
   while a short-lived :class:`BindingSession` owns the DuckDB replacement
   scans and cursors backing those names.

Option 1 avoids duplicating graph nodes but requires changing the compiler's
public translation signature, plumbing context through every translation
rule and keeping a mutable side-channel in sync with the graph. It also
makes "compile without registering anything" awkward, since compilation and
execution would have to reconstruct the exact same name assignment
independently.

Option 2 keeps the compiler untouched: the rewrite is a standard,
non-mutating ``Node.replace`` call (the caller's expression and mapping are
never modified), identity semantics fall out of structural node equality
for free, and compile/execute necessarily agree on names because both run
the exact same pure function. Its costs are a one-time copy of the graph's
node spine (the shared subgraph is still shared, so the cost is linear in
the number of nodes) and a small backend-local operation class.

Relation names are derived from the placeholder node's precomputed
structural hash, so identical placeholders compile to the same name within a
process (and compile/execute always agree) while same-named/different-schema
placeholders never collide; a UUID name is used only in the theoretical
event of an in-mapping hash collision.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import pyarrow as pa

import ibis.common.exceptions as exc
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
from ibis import util
from ibis.expr.schema import Schema  # noqa: TC001 - used at class runtime

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    import ibis.expr.types as ir
    from ibis.backends.duckdb import Backend


class BoundRelation(ops.PhysicalTable):
    """A physical-table node backed by a per-call arrow binding.

    The node only records the generated relation name and the declared
    schema; the actual ``pyarrow.Table`` data is held by the
    :class:`BindingSession` that created the node, so that expression graphs
    stay free of large data payloads and remain cheap to copy and hash.
    """

    schema: Schema


# The ibis types accepted as relation-binding columns, mapped to the exact
# pyarrow type that must be presented. No implicit conversion is performed.
_SUPPORTED_TYPE_KINDS = (
    dt.Int64,
    dt.Float64,
    dt.String,
    dt.Boolean,
    dt.Timestamp,
)


def _expected_arrow_type(dtype: dt.DataType) -> pa.DataType:
    """Return the exact pyarrow type accepted for an ibis column type."""
    if isinstance(dtype, dt.Timestamp):
        # only microsecond precision, timezone-naive timestamps are accepted
        if dtype.timezone is not None or dtype.scale not in (None, 6):
            raise exc.IbisInputError(
                "only microsecond precision, timezone-naive timestamps are "
                f"supported, got {dtype}"
            )
        return pa.timestamp("us")
    return dtype.to_pyarrow()


def _describe_table(node: ops.UnboundTable) -> str:
    return f"unbound table {node.name!r}"


def _validate_binding(
    node: ops.UnboundTable, arrow_table: pa.Table
) -> dict[str, pa.ChunkedArray]:
    """Validate a single arrow table against its placeholder's schema.

    Returns the arrow columns keyed by declared column name, reordered to
    match the declared schema. Reordering is zero copy; the caller's arrow
    table is never mutated.
    """
    label = _describe_table(node)

    if not isinstance(arrow_table, pa.Table):
        raise exc.IbisInputError(
            f"value bound to {label} must be a pyarrow.Table, got "
            f"{type(arrow_table).__name__}"
        )

    declared = node.schema
    actual_names = set(arrow_table.column_names)
    declared_names = set(declared.names)

    if missing := declared_names - actual_names:
        cols = ", ".join(sorted(missing))
        raise exc.IbisInputError(f"binding for {label} is missing column(s): {cols}")
    if extra := actual_names - declared_names:
        cols = ", ".join(sorted(extra))
        raise exc.IbisInputError(
            f"binding for {label} has unexpected column(s): {cols}"
        )

    validated: dict[str, pa.ChunkedArray] = {}
    for col_name, ibis_type in declared.items():
        if not isinstance(ibis_type, _SUPPORTED_TYPE_KINDS):
            raise exc.IbisInputError(
                f"column {col_name!r} of {label} has unsupported type "
                f"{ibis_type}; supported types are int64, float64, string, "
                "boolean and microsecond timezone-naive timestamp"
            )

        column = arrow_table.column(col_name)
        expected = _expected_arrow_type(ibis_type)
        if column.type != expected:
            raise exc.IbisInputError(
                f"column {col_name!r} of {label} declares {ibis_type} "
                f"(arrow type {expected}) but the binding provides "
                f"arrow type {column.type}; no implicit conversion is applied"
            )

        if not ibis_type.nullable and column.null_count:
            raise exc.IbisInputError(
                f"column {col_name!r} of {label} is declared non-nullable "
                f"but its binding contains {column.null_count} null value(s)"
            )

        validated[col_name] = column

    return validated


def _normalize_bindings(
    table_bindings: Mapping[ir.Table, pa.Table],
) -> dict[ops.UnboundTable, pa.Table]:
    """Normalize an expression-keyed mapping to an operation-keyed mapping."""
    normalized: dict[ops.UnboundTable, pa.Table] = {}
    for key, value in table_bindings.items():
        node = getattr(key, "op", None)
        node = node() if callable(node) else node
        if not isinstance(node, ops.UnboundTable):
            raise exc.IbisInputError(
                "table_bindings keys must be unbound Table expressions "
                f"created with ibis.table(); got {type(key).__name__}"
            )
        if node in normalized:
            raise exc.IbisInputError(
                f"duplicate binding provided for {_describe_table(node)}"
            )
        normalized[node] = value
    return normalized


class BindingSession:
    """Owns the DuckDB resources created for one binding-enabled call.

    A session is responsible for:

    * validating every binding before any catalog resource is created;
    * assigning a relation name to each bound placeholder;
    * rewriting the expression graph so that placeholders are replaced by
      :class:`BoundRelation` nodes;
    * registering the arrow data on a dedicated DuckDB child connection (or
      on the backend connection for eager calls), where it shadows any
      same-named catalog table only for that connection;
    * unregistering everything and closing cursors when the call finishes,
      including error paths, without closing the backend itself.
    """

    def __init__(
        self,
        backend: Backend,
        expr: ir.Expr,
        table_bindings: Mapping[ir.Table, pa.Table],
    ) -> None:
        self._backend = backend

        self._name_by_node: dict[ops.UnboundTable, str] = {}
        self._data_by_name: dict[str, pa.Table] = {}
        self._registered: list[str] = []
        self._cursors: list[Any] = []
        self._closed = False

        # All validation happens here, before a single DuckDB catalog call.
        self._normalized = _normalize_bindings(table_bindings)
        self._validate_completeness(expr)
        self._check_database_tables(expr, backend)
        self._bound_expr = self._rewrite(expr)

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def _validate_completeness(self, expr: ir.Expr) -> None:
        unbound = expr.op().find(ops.UnboundTable)
        seen: set[ops.UnboundTable] = set()
        missing: set[ops.UnboundTable] = set()
        for node in unbound:
            if node in seen:
                continue
            seen.add(node)
            if node not in self._normalized:
                missing.add(node)
        if missing:
            names = ", ".join(sorted(_describe_table(node) for node in missing))
            raise exc.IbisInputError(
                "table_bindings is a non-empty mapping and therefore must "
                f"provide data for every unbound table; missing binding(s) "
                f"for: {names}"
            )
        if extra := set(self._normalized) - seen:
            names = ", ".join(sorted(_describe_table(node) for node in extra))
            raise exc.IbisInputError(
                "table_bindings contains key(s) that do not appear in the "
                f"expression: {names}"
            )

    @staticmethod
    def _check_database_tables(expr: ir.Expr, backend: Backend) -> None:
        """Reject bound tables belonging to a different backend."""
        for node in expr.op().find(ops.DatabaseTable):
            if node.source is not backend:
                raise exc.IbisError(
                    "expression contains a database table bound to a "
                    f"different {node.source.name!r} backend instance than "
                    "the backend requested for execution; cross-backend "
                    "table copying is not performed for relation bindings"
                )

    def _rewrite(self, expr: ir.Expr) -> ir.Expr:
        replacement: dict[ops.Node, ops.Node] = {}
        # de-duplicate structurally equal placeholders: every reference to
        # the same node (including self-joined views) shares one relation
        used_names: set[str] = set()
        for node, arrow_table in self._normalized.items():
            # validates columns/types/nullability before naming anything,
            # and returns columns in declared order (zero-copy reorder)
            validated_columns = _validate_binding(node, arrow_table)

            # derive the relation name from the node's structural identity
            # (not its table name): identical placeholder nodes compile to
            # the same name across calls, while same-named/different-schema
            # placeholders get different names. Guard against the
            # theoretical hash collision with a UUID fallback.
            name = f"ibis_duckdb_binding_{hash(node) & ((1 << 64) - 1):016x}"
            if name in used_names:
                name = util.gen_name("duckdb_binding")
            used_names.add(name)

            self._name_by_node[node] = name
            # rebuild with the declared column order; the caller's arrow
            # table is left untouched
            ordered = pa.table(dict(validated_columns))
            self._data_by_name[name] = ordered

            replacement[node] = BoundRelation(name=name, schema=node.schema)

        new_op = expr.op().replace(replacement)
        return expr.__class__(new_op)

    # ------------------------------------------------------------------
    # public accessors
    # ------------------------------------------------------------------
    @property
    def expr(self) -> ir.Expr:
        """The rewritten expression ready for normal compilation."""
        return self._bound_expr

    def register_all(self, con: Any) -> None:
        """Register every bound arrow table on ``con``.

        Registration is transactional from the caller's point of view: a
        failure halfway through rolls back the names already registered, so
        a failed call never leaves partial bindings behind.
        """
        try:
            for name, data in self._data_by_name.items():
                con.register(name, data)
                self._registered.append(name)
        except BaseException:
            self._unregister_from(con)
            raise

    def track_cursor(self, cursor: Any) -> Any:
        """Record a cursor whose lifetime is tied to this session."""
        self._cursors.append(cursor)
        return cursor

    # ------------------------------------------------------------------
    # teardown
    # ------------------------------------------------------------------
    def _unregister_from(self, con: Any) -> None:
        while self._registered:
            name = self._registered.pop()
            with contextlib.suppress(Exception):
                con.unregister(name)

    def close(self) -> None:
        """Release every resource created for this call.

        Idempotent. Never closes the backend's own connection; only child
        cursors created by the session are closed.
        """
        if self._closed:
            return
        self._closed = True
        # names may live on different child connections, so unregister on
        # every connection we touched
        for con in self._connections:
            self._unregister_from(con)
        while self._cursors:
            cursor = self._cursors.pop()
            with contextlib.suppress(Exception):
                cursor.close()
        # drop our references to caller-owned arrow data
        self._data_by_name.clear()
        self._normalized.clear()
        self._name_by_node.clear()

    @property
    def _connections(self) -> Iterator[Any]:
        seen: set[int] = set()
        for cursor in self._cursors:
            if id(cursor) not in seen:
                seen.add(id(cursor))
                yield cursor
        backend_con = self._backend.con
        if id(backend_con) not in seen:
            yield backend_con

    @property
    def registered_names(self) -> tuple[str, ...]:
        """Names currently registered by this session (test/introspection)."""
        return tuple(self._registered)

    @property
    def holds_data(self) -> bool:
        """Whether the session still references caller arrow data."""
        return bool(self._data_by_name)

    @property
    def active_cursors(self) -> tuple[Any, ...]:
        """Child cursors the session still holds open."""
        return tuple(self._cursors)

    def __enter__(self) -> BindingSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@contextlib.contextmanager
def eager_binding(
    backend: Backend,
    expr: ir.Expr,
    table_bindings: Mapping[ir.Table, pa.Table] | None,
) -> Iterator[ir.Expr | None]:
    """Bind relations for an eager call.

    Yields the rewritten expression (or ``None`` when no bindings are
    requested). All registrations happen on the backend connection itself
    and are torn down on exit -- again, only after successful validation.
    """
    if not table_bindings:
        yield None
        return

    session = BindingSession(backend, expr, table_bindings)
    try:
        session.register_all(backend.con)
        yield session.expr
    finally:
        session.close()


class BindingReader:
    """A record-batch reader that owns a :class:`BindingSession`.

    DuckDB replacement scans are connection local, so streaming sessions
    register their data on a dedicated child connection (a cursor) which
    is kept open for the lifetime of the stream. Closing, exhausting or
    erroring out of the reader unregisters the names, closes the cursor and
    drops references to the caller's arrow tables -- deterministically,
    without waiting for garbage collection and without buffering the full
    result in Python.

    This is intentionally a standalone wrapper rather than a subclass of
    ``pa.ipc.RecordBatchReader`` (pyarrow's reader is a C-extension type
    whose direct subclassing segfaults); it mirrors the public iteration,
    context-manager and ``read_*`` surface of a real reader.
    """

    def __init__(
        self,
        inner: pa.ipc.RecordBatchReader,
        session: BindingSession,
        cursor: Any,
    ) -> None:
        self._inner = inner
        self._session = session
        self._cursor = cursor
        self._closed = False

    @property
    def schema(self) -> pa.Schema:
        return self._inner.schema

    def read_next_batch(self) -> pa.RecordBatch:
        try:
            return self._inner.read_next_batch()
        except BaseException:
            # includes normal StopIteration (exhaustion) and real errors
            self.close()
            raise

    def read_next_batch_with_custom_metadata(self):
        try:
            return self._inner.read_next_batch_with_custom_metadata()
        except BaseException:
            self.close()
            raise

    def read_all(self) -> pa.Table:
        try:
            return self._inner.read_all()
        finally:
            self.close()

    def read_pandas(self):
        try:
            return self._inner.read_pandas()
        finally:
            self.close()

    def cast(self, target_schema):
        # return a plain pyarrow reader; session teardown is triggered now
        # since all batches are consumed during the cast
        try:
            return self._inner.cast(target_schema)
        finally:
            self.close()

    def iter_batches_with_custom_metadata(self):
        while not self._closed:
            try:
                yield self._inner.read_next_batch_with_custom_metadata()
            except StopIteration:
                self.close()
                return
            except BaseException:
                self.close()
                raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._inner.close()
        self._session.close()

    def __iter__(self) -> Iterator[pa.RecordBatch]:
        # one iterator per reader, mirroring pa.ipc.RecordBatchReader
        # semantics; the session is released on normal exhaustion, on
        # explicit close or on an error -- never merely because an
        # iterator object is dropped before being advanced
        while not self._closed:
            try:
                yield self._inner.read_next_batch()
            except StopIteration:
                self.close()
                return
            except BaseException:
                self.close()
                raise

    def __enter__(self) -> BindingReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        # safety net only: explicit close / normal exhaustion are the
        # contract; this avoids resource retention if a caller drops the
        # reader without draining it
        self.close()
