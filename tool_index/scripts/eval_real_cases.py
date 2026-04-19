"""Evaluate recall@k against the hand-written natural-language test cases.

Reads SIMPLE_CASES + COMPLEX_CASES from data/generateTools/test_cases.py.
Each case has a query string and a list of expected tool calls. We map
each tool function (e.g. json_format) to its descriptor.id and then ask
the productized D3+D1 retriever (multi-vector traverser + ColBERT-style
rerank) whether the gold tools appear in the top-k.

Two metrics:
  - simple recall@k: fraction of single-call cases where the one expected
    tool is in top-k.
  - complex set-recall@k: fraction of expected tools across all complex
    cases that appear in top-k for that case (micro-averaged).

Usage:
    uv run scripts/eval_real_cases.py --run raw-tools --k 10
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

from _pipeline_config import make_config, run_dir
from tool_index.retrieval import precompute_tool_vectors, retrieve, rerank_tools
from tool_index.schema import Enrichment, Tree, ToolDescriptor


def _load_test_cases(repo_root: Path) -> tuple[list[dict], list[dict]]:
    # tools is already loaded by _pipeline_config under the
    # "generateTools" package alias; reuse it so the relative import in
    # test_cases.py resolves.
    spec = importlib.util.spec_from_file_location(
        "generateTools.test_cases", repo_root / "data/generateTools/test_cases.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["generateTools.test_cases"] = mod
    spec.loader.exec_module(mod)
    return mod.SIMPLE_CASES, mod.COMPLEX_CASES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rerank-k", type=int, default=20)
    ap.add_argument("--beam", type=int, default=3)
    args = ap.parse_args()

    cfg = make_config()
    out = run_dir(args.run)
    repo_root = Path(__file__).parent.parent

    descriptors = [ToolDescriptor.from_dict(d) for d in json.loads((out / "01_descriptors.json").read_text())]
    name_to_id = {d.name: d.id for d in descriptors}

    enrichments = {tid: Enrichment.from_dict(d) for tid, d in json.loads((out / "02_enrichments.json").read_text()).items()}
    tree_payload = json.loads((out / "tree_draft.json").read_text())
    tree_payload.pop("_meta", None)
    tree = Tree.from_dict(tree_payload)

    tool_vectors = precompute_tool_vectors(enrichments, cfg.embedder)
    print(f"Loaded {len(descriptors)} tools, {len(tool_vectors)} tool vector sets")

    simple_cases, complex_cases = _load_test_cases(repo_root)
    print(f"Test set: {len(simple_cases)} simple + {len(complex_cases)} complex = {len(simple_cases)+len(complex_cases)} cases")

    def topk_for(query: str) -> list[str]:
        q_emb = cfg.embedder.embed(query)
        candidates = retrieve(tree, q_emb, k=args.rerank_k, beam=args.beam)
        return rerank_tools(q_emb, candidates, tool_vectors, args.k)

    # Simple recall@k.
    simple_hits = 0
    simple_seen = 0
    simple_unmapped: list[str] = []
    simple_misses: list[tuple[str, str]] = []
    for case in simple_cases:
        gold_name = case["calls"][0]
        gold_id = name_to_id.get(gold_name)
        if gold_id is None:
            simple_unmapped.append(gold_name)
            continue
        simple_seen += 1
        topk = topk_for(case["query"])
        if gold_id in topk:
            simple_hits += 1
        else:
            simple_misses.append((case["query"], gold_name))

    # Complex set-recall@k (micro-average over expected tools).
    complex_total = 0
    complex_found = 0
    complex_misses: list[tuple[str, list[str]]] = []
    for case in complex_cases:
        gold_names = case["calls"]
        gold_ids = [name_to_id[n] for n in gold_names if n in name_to_id]
        if not gold_ids:
            continue
        topk = set(topk_for(case["query"]))
        found = sum(1 for gid in gold_ids if gid in topk)
        complex_total += len(gold_ids)
        complex_found += found
        if found < len(gold_ids):
            missing = [gold_names[i] for i, gid in enumerate(gold_ids) if gid not in topk]
            complex_misses.append((case["query"], missing))

    print()
    print(f"SIMPLE recall@{args.k}: {simple_hits}/{simple_seen} = {simple_hits/simple_seen:.3f}  (unmapped tool names: {len(simple_unmapped)})")
    print(f"COMPLEX set-recall@{args.k}: {complex_found}/{complex_total} = {complex_found/complex_total:.3f}")
    print(f"COMPLEX cases with all gold present: {len(complex_cases)-len(complex_misses)}/{len(complex_cases)}")
    if simple_unmapped:
        print(f"\nSimple unmapped tool names ({len(simple_unmapped)}): {sorted(set(simple_unmapped))}")

    if simple_misses:
        print(f"\nSimple misses ({len(simple_misses)}):")
        for q, gold in simple_misses[:10]:
            print(f"  - {gold!r:48s} <- {q[:80]}")

    if complex_misses:
        print(f"\nComplex partial misses ({len(complex_misses)}):")
        for q, missing in complex_misses[:10]:
            print(f"  - missing {missing} <- {q[:80]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
