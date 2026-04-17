"""Stable string constants for the five tree levels.

The index is a fixed 4-level hierarchy plus a synthetic root:

    L0 root
    └── L1 domain       (e.g. "data & persistence")
        └── L2 category (e.g. "relational storage")
            └── L3 group (e.g. "user table CRUD")
                └── L4 tool  (the actual callable leaf)

Level strings are persisted in snapshots, so **do not rename them** without a
snapshot schema bump in `storage/versioning.py`. `LEVEL_ORDER` is the canonical
top-down traversal order used by `retrieval/traverser.py` and validation.
"""

# Synthetic single root node that owns every L1 domain. Not model-facing.
LEVEL_ROOT = "L0"

# Top-level buckets (broad competency areas). Typical fanout 2-20.
LEVEL_DOMAIN = "L1"

# Mid-level groupings inside a domain. Typical fanout 2-10.
LEVEL_CATEGORY = "L2"

# Fine-grained functional cluster; direct parent of tool leaves. Typical fanout 2-10.
LEVEL_GROUP = "L3"

# Actual tool descriptors (leaves). Fanout per group is typically 3-8.
LEVEL_TOOL = "L4"

# Canonical top-down order. Index i is the parent of index i+1.
LEVEL_ORDER = [LEVEL_ROOT, LEVEL_DOMAIN, LEVEL_CATEGORY, LEVEL_GROUP, LEVEL_TOOL]

# Allowed values for `ToolDescriptor.side_effect_class`. Used by validators
# and (future) routing heuristics that want to know whether a tool mutates state.
#   read      — pure lookup, idempotent
#   write     — mutates state
#   transact  — multi-step commit semantics (e.g. payments)
#   compute   — in-memory computation, no external I/O
SIDE_EFFECTS = {"read", "write", "transact", "compute"}
