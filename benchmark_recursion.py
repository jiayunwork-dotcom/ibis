"""Performance characterization for ``Table.recurse``.

Run with a backend URL, e.g.::

    python benchmark_recursion.py duckdb://
    python benchmark_recursion.py polars://

The workload is a chain of ``n`` child->parent edges rooted at a single seed,
so the fixed point needs exactly ``n`` productive rounds plus one empty
application. The same plan graph is used at every budget (it is never unrolled
per iteration), and the metrics below record that explicitly.
"""

from __future__ import annotations

import resource
import sys
import time

import ibis
import ibis.expr.operations as ops


def run(backend: str, n: int) -> dict:
    con = ibis.connect(backend)
    edges = ibis.memtable({"child": list(range(1, n)), "parent": list(range(n - 1))})
    root = ibis.memtable({"node": [n - 1]})
    expr = root.recurse(
        lambda k: k.join(edges, k.node == edges.child).select(node=edges.parent),
        max_iterations=n + 5,
    )

    # the IR plan size is constant in n (the graph is not unrolled)
    plan_nodes = len(expr.op().find(ops.Node))

    # number of candidate rounds actually processed (the chain needs n
    # productive rounds + one empty application), tracked independently of the
    # backend by the graph's structure
    expected_rounds = n + 1

    t0 = time.time()
    result = con.to_pyarrow(expr)
    elapsed = time.time() - t0
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    return {
        "backend": backend,
        "n": n,
        "result_rows": result.num_rows,
        "expected_rounds": expected_rounds,
        "plan_nodes": plan_nodes,
        "seconds": round(elapsed, 3),
        "peak_rss_mb": round(peak_mb, 1),
        "ms_per_round": round(1000 * elapsed / expected_rounds, 3),
    }


def main() -> None:
    backends = sys.argv[1:] or ["duckdb://", "polars://"]
    rows = []
    for backend in backends:
        for n in (1000, 10000):
            rows.append(run(backend, n))
    widths = (max(len(str(r[k])) for r in rows) for k in rows[0])
    header = "  ".join(str(k).ljust(w) for k, w in zip(rows[0], widths))
    print(header)
    for r in rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(r.values(), widths)))


if __name__ == "__main__":
    main()
