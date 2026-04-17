"""Serialization format hooks.

Currently a placeholder — all formats are plain JSON, handled directly by
``snapshot.py`` and the ``to_dict`` / ``from_dict`` methods on schema
types. Kept as a module so we have somewhere to add an alternative
format (parquet for embeddings, msgpack for trees, etc.) without
restructuring callers.
"""
# JSON-only formats. Parquet optional; intentionally omitted to keep deps tiny.
