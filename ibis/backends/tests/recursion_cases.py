"""Shared fixtures and hand-computed expectations for recurse backends.

The expected sets are computed by hand from small, fully specified relations
rather than produced by another executor, so the tests are genuine oracles.
"""

from __future__ import annotations

import ibis


def ancestor_edges():
    # child -> parent, with a diamond (1 reaches 3 two ways), a duplicate
    # edge (4 -> 5 twice) and a cycle (5 -> 2)
    return ibis.memtable(
        {
            "child": [1, 1, 2, 3, 4, 4, 5],
            "parent": [2, 3, 3, 4, 5, 5, 2],
        }
    )


def ancestor_seed():
    return ibis.memtable({"node": [1]})


def ancestor_step(edges):
    def step(known):
        return known.join(edges, known.node == edges.child).select(node=edges.parent)

    return step


# Hand-traced closure from seed {1}:
#   round 0: {1}
#   round 1: parents of 1 = {2, 3}
#   round 2: parents of 2,3 = {3(seen), 4}          -> add {4}
#   round 3: parents of 4 = {5}                     -> add {5}
#   round 4: parents of 5 = {2(seen)}               -> add nothing (converge)
ANCESTOR_CLOSURE = [1, 2, 3, 4, 5]
# productive rounds = 4 (depths 0..3), plus one empty application = 5 total
ANCESTOR_APPLICATIONS = 5


def chain_seed_single():
    return ibis.memtable({"x": [0]})


def chain_step():
    # x -> x+1 while x <= 5
    return lambda k: k.mutate(x=k.x + 1).filter(lambda d: d.x <= 5)


CHAIN_SINGLE_CLOSURE = [0, 1, 2, 3, 4, 5]
# five productive applications (add 1,2,3,4,5) plus the empty one = 6
CHAIN_SINGLE_APPLICATIONS = 6


def multi_root_seed():
    return ibis.memtable({"x": [0, 1, 2, 3, 4]})


MULTI_ROOT_CLOSURE = [0, 1, 2, 3, 4, 5]
# round 1 adds only {5}; round 2 is empty -> 2 applications
MULTI_ROOT_APPLICATIONS = 2


def null_rows():
    return ibis.memtable(
        {
            "a": [1, 2, None, 1],
            "b": ["x", "y", None, "x"],
            "c": [True, False, True, True],
        },
        schema={"a": "int64", "b": "string", "c": "boolean"},
    )


def diamond_edges():
    # 1 -> {2,3}; 2 -> {4,5}; 3 -> 4. Node 4 is reached on two distinct paths
    # in the same round, exercising per-round distinctness.
    return ibis.memtable({"child": [1, 1, 2, 3, 2], "parent": [2, 3, 4, 4, 5]})


def diamond_seed():
    return ibis.memtable({"node": [1]})


# Hand-traced rounds:
#   r0 {1}; r1 adds {2,3}; r2 adds {4,5}; r3 adds nothing (converge)
DIAMOND_CLOSURE = [1, 2, 3, 4, 5]
DIAMOND_APPLICATIONS = 3


# distinct rows of null_rows: NULL in the same (a,b) position counts as equal
NULL_DISTINCT_COUNT = 3
NULL_DISTINCT_SORTED = [
    (1, "x", True),
    (2, "y", False),
    (None, None, True),
]
