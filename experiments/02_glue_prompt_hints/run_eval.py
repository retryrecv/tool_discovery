"""Run eval with the glue-tool-hinted decomposer prompt.

Mirrors `tool_index/scripts/eval_real_cases.py` but imports
`decompose_query` from `decompose_with_hints` instead of the stock
`tool_index.retrieval`. Everything else (retrieval, rerank, tree,
embeddings, beam, k) is identical to production.

Outputs:
  - Console summary with simple/complex/ultra metrics + miss list
  - decompositions.jsonl with every (query, sub_queries) pair for
    qualitative inspection

Usage:
    cd tool_index/
    uv run ../experiments/02_glue_prompt_hints/run_eval.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TOOL_INDEX = REPO_ROOT / "tool_index"
SPIKE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(TOOL_INDEX / "scripts"))
sys.path.insert(0, str(SPIKE_DIR))

from _pipeline_config import make_config, run_dir  # noqa: E402

# Hinted decomposer (replaces tool_index.retrieval.decompose_query)
from decompose_with_hints import decompose_query  # noqa: E402

from tool_index.retrieval import (  # noqa: E402
    precompute_tool_vectors,
    retrieve_decomposed,
)
from tool_index.schema import Enrichment, Tree, ToolDescriptor  # noqa: E402


RUN = "raw-tools"
K = 10
RERANK_K = 20
BEAM = 3


def _load_eval_queries() -> tuple[list[dict], list[dict], list[dict]]:
    spec = importlib.util.spec_from_file_location(
        "corpus.eval_queries",
        TOOL_INDEX / "data/corpus/eval_queries.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["corpus.eval_queries"] = mod
    spec.loader.exec_module(mod)
    return (
        mod.SIMPLE_CASES,
        mod.COMPLEX_CASES,
        getattr(mod, "ULTRA_COMPLEX_CASES", []),
    )


def _summarize_node(desc: str, max_chars: int = 140) -> str:
    s = desc.strip().replace("\n", " ")
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "..."


def _schema_lines(tree: Tree) -> list[str]:
    l1 = [tree.nodes_by_id[cid] for cid in tree.root.children if cid in tree.nodes_by_id]
    chosen = l1 if len(l1) > 1 else [
        tree.nodes_by_id[ccid]
        for n in l1
        for ccid in n.children
        if ccid in tree.nodes_by_id
    ]
    return [_summarize_node(n.description) for n in chosen]


def _set_recall(cases, name_to_id, topk_for):
    total = 0
    found_total = 0
    misses: list[tuple[str, list[str]]] = []
    for case in cases:
        gold_names = case["calls"]
        gold_ids = [name_to_id[n] for n in gold_names if n in name_to_id]
        if not gold_ids:
            continue
        topk = set(topk_for(case["query"]))
        found = sum(1 for gid in gold_ids if gid in topk)
        total += len(gold_ids)
        found_total += found
        if found < len(gold_ids):
            missing = [gold_names[i] for i, gid in enumerate(gold_ids) if gid not in topk]
            misses.append((case["query"], missing))
    return found_total, total, misses


def main() -> int:
    cfg = make_config()
    out = run_dir(RUN)

    descriptors = [
        ToolDescriptor.from_dict(d)
        for d in json.loads((out / "01_descriptors.json").read_text())
    ]
    name_to_id = {d.name: d.id for d in descriptors}

    enrichments = {
        tid: Enrichment.from_dict(d)
        for tid, d in json.loads((out / "02_enrichments.json").read_text()).items()
    }
    tree_payload = json.loads((out / "tree_draft.json").read_text())
    tree_payload.pop("_meta", None)
    tree = Tree.from_dict(tree_payload)

    tool_vectors = precompute_tool_vectors(enrichments, cfg.embedder)
    print(f"Loaded {len(descriptors)} tools, {len(tool_vectors)} tool vector sets")

    schema_lines = _schema_lines(tree)
    print(f"Decomposition ON (HINTED): injecting {len(schema_lines)} schema lines")

    simple_cases, complex_cases, ultra_complex_cases = _load_eval_queries()
    print(
        f"Test set: {len(simple_cases)} simple + {len(complex_cases)} complex + "
        f"{len(ultra_complex_cases)} ultra-complex"
    )
    print(f"Pipeline: beam={BEAM}, rerank_k={RERANK_K}, k={K}")
    print()

    decompose_log: list[dict] = []

    def topk_for(query: str) -> list[str]:
        sub_queries = decompose_query(query, cfg.enricher_llm, schema_lines)
        decompose_log.append({"query": query, "sub_queries": sub_queries})
        sub_embeddings = [cfg.embedder.embed(sq) for sq in sub_queries]
        return retrieve_decomposed(
            tree, sub_embeddings, tool_vectors,
            k=K, rerank_k=RERANK_K, beam=BEAM,
        )

    simple_hits = 0
    simple_seen = 0
    simple_misses: list[tuple[str, str]] = []
    for case in simple_cases:
        gold_name = case["calls"][0]
        gold_id = name_to_id.get(gold_name)
        if gold_id is None:
            continue
        simple_seen += 1
        topk = topk_for(case["query"])
        if gold_id in topk:
            simple_hits += 1
        else:
            simple_misses.append((case["query"], gold_name))

    complex_found, complex_total, complex_misses = _set_recall(complex_cases, name_to_id, topk_for)
    ultra_found, ultra_total, ultra_misses = _set_recall(ultra_complex_cases, name_to_id, topk_for)

    print(
        f"SIMPLE recall@{K}: {simple_hits}/{simple_seen} = {simple_hits/simple_seen:.3f}"
    )
    print(
        f"COMPLEX set-recall@{K}: {complex_found}/{complex_total} = {complex_found/complex_total:.3f}"
    )
    print(
        f"COMPLEX full-cover: {len(complex_cases)-len(complex_misses)}/{len(complex_cases)} = "
        f"{(len(complex_cases)-len(complex_misses))/len(complex_cases):.3f}"
    )
    if ultra_complex_cases:
        print(
            f"ULTRA  set-recall@{K}: {ultra_found}/{ultra_total} = {ultra_found/ultra_total:.3f}"
        )
        print(
            f"ULTRA  full-cover: {len(ultra_complex_cases)-len(ultra_misses)}/{len(ultra_complex_cases)} = "
            f"{(len(ultra_complex_cases)-len(ultra_misses))/len(ultra_complex_cases):.3f}"
        )

    print()
    print("Production baseline (LLM decomposer w/o hints, raw-tools/v8, 2026-05-01):")
    print("  SIMPLE recall@10        : 0.960")
    print("  COMPLEX set-recall@10   : 0.908")
    print("  COMPLEX full-cover      : 0.800")
    print()

    if simple_misses:
        print(f"SIMPLE misses ({len(simple_misses)}):")
        for q, gold in simple_misses[:10]:
            print(f"  - {gold!r:55s} <- {q}")

    if complex_misses:
        print(f"\nCOMPLEX partial misses ({len(complex_misses)}):")
        for q, gold in complex_misses[:15]:
            print(f"  - missing {gold} <- {q}")

    if ultra_complex_cases and ultra_misses:
        print(f"\nULTRA partial misses ({len(ultra_misses)}):")
        for q, gold in ultra_misses[:15]:
            print(f"  - missing {gold} <- {q}")

    decomp_path = SPIKE_DIR / "decompositions.jsonl"
    with decomp_path.open("w") as f:
        for entry in decompose_log:
            f.write(json.dumps(entry) + "\n")
    avg_subq = sum(len(e["sub_queries"]) for e in decompose_log) / max(1, len(decompose_log))
    split_count = sum(1 for e in decompose_log if len(e["sub_queries"]) > 1)
    print(f"\nDecomposition stats: {split_count}/{len(decompose_log)} queries split; avg sub-queries = {avg_subq:.2f}")
    print(f"Wrote {len(decompose_log)} decompositions to {decomp_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
