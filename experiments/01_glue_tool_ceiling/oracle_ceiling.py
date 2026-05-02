"""Oracle-decomposition ceiling for complex cases.

Bypasses the LLM decomposer and feeds `retrieve_decomposed` an oracle
list of sub-queries built from each gold tool's `intent_phrase`. This
upper-bounds what the current retrieval+rerank stack can achieve given
a perfect decomposition.

Compare the oracle metrics to the production eval (LLM decomposition):

    Production (eval_real_cases --decompose, raw-tools/v8, 2026-05-01):
        SIMPLE recall@10        : 0.960
        COMPLEX set-recall@10   : 0.908
        COMPLEX full-cover      : 0.800

If oracle complex recall ≈ 1.0 → the gap is decomposition-side.
If oracle complex recall < 0.95 → the gap also has a retrieval residue.

Usage:
    uv run experiments/01_glue_tool_ceiling/oracle_ceiling.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TOOL_INDEX = REPO_ROOT / "tool_index"
sys.path.insert(0, str(TOOL_INDEX / "scripts"))

# Importing _pipeline_config has side effects: loads .env, registers the
# `corpus.catalog` module, and exposes `raw_tools` / `make_config`.
from _pipeline_config import make_config, run_dir  # noqa: E402

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


def _set_recall(
    cases: list[dict],
    name_to_id: dict[str, str],
    topk_for,
) -> tuple[int, int, list[tuple[str, list[str]]]]:
    total = 0
    found_total = 0
    misses: list[tuple[str, list[str]]] = []
    for case in cases:
        gold_names = case["calls"]
        gold_ids = [name_to_id[n] for n in gold_names if n in name_to_id]
        if not gold_ids:
            continue
        topk = set(topk_for(case))
        found = sum(1 for gid in gold_ids if gid in topk)
        total += len(gold_ids)
        found_total += found
        if found < len(gold_ids):
            missing = [
                gold_names[i] for i, gid in enumerate(gold_ids) if gid not in topk
            ]
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
    id_to_name = {d.id: d.name for d in descriptors}

    enrichments = {
        tid: Enrichment.from_dict(d)
        for tid, d in json.loads((out / "02_enrichments.json").read_text()).items()
    }
    tree_payload = json.loads((out / "tree_draft.json").read_text())
    tree_payload.pop("_meta", None)
    tree = Tree.from_dict(tree_payload)

    tool_vectors = precompute_tool_vectors(enrichments, cfg.embedder)
    print(f"Loaded {len(descriptors)} tools, {len(tool_vectors)} tool vector sets")

    simple_cases, complex_cases, ultra_complex_cases = _load_eval_queries()
    print(
        f"Test set: {len(simple_cases)} simple + {len(complex_cases)} complex + "
        f"{len(ultra_complex_cases)} ultra-complex"
    )
    print(f"Mode: ORACLE decomposition (sub-queries = gold tools' intent_phrases)")
    print(f"Pipeline: beam={BEAM}, rerank_k={RERANK_K}, k={K}")
    print()

    def oracle_topk(case: dict) -> list[str]:
        gold_names = case["calls"]
        gold_ids = [name_to_id[n] for n in gold_names if n in name_to_id]
        sub_queries = []
        for gid in gold_ids:
            enr = enrichments.get(gid)
            if enr is not None and enr.intent_phrase:
                sub_queries.append(enr.intent_phrase)
            else:
                sub_queries.append(id_to_name[gid].replace("_", " "))
        sub_embeddings = [cfg.embedder.embed(sq) for sq in sub_queries]
        return retrieve_decomposed(
            tree,
            sub_embeddings,
            tool_vectors,
            k=K,
            rerank_k=RERANK_K,
            beam=BEAM,
        )

    complex_found, complex_total, complex_misses = _set_recall(
        complex_cases, name_to_id, oracle_topk
    )
    ultra_found, ultra_total, ultra_misses = _set_recall(
        ultra_complex_cases, name_to_id, oracle_topk
    )

    print(
        f"COMPLEX set-recall@{K} (oracle):   "
        f"{complex_found}/{complex_total} = {complex_found/complex_total:.3f}"
    )
    print(
        f"COMPLEX full-cover (oracle):       "
        f"{len(complex_cases)-len(complex_misses)}/{len(complex_cases)} = "
        f"{(len(complex_cases)-len(complex_misses))/len(complex_cases):.3f}"
    )
    if ultra_complex_cases:
        print(
            f"ULTRA  set-recall@{K} (oracle):    "
            f"{ultra_found}/{ultra_total} = {ultra_found/ultra_total:.3f}"
        )
        print(
            f"ULTRA  full-cover (oracle):        "
            f"{len(ultra_complex_cases)-len(ultra_misses)}/{len(ultra_complex_cases)} = "
            f"{(len(ultra_complex_cases)-len(ultra_misses))/len(ultra_complex_cases):.3f}"
        )

    print()
    print("Production baseline (LLM decomposition, raw-tools/v8, 2026-05-01):")
    print("  COMPLEX set-recall@10 : 0.908")
    print("  COMPLEX full-cover    : 0.800")
    print()

    if complex_misses:
        print(f"COMPLEX residual misses ({len(complex_misses)}):")
        for q, gold in complex_misses[:15]:
            print(f"  - missing {gold} <- {q}")
    else:
        print("COMPLEX: zero residual misses — entire gap is decomposition-side.")

    if ultra_complex_cases and ultra_misses:
        print(f"\nULTRA residual misses ({len(ultra_misses)}):")
        for q, gold in ultra_misses[:15]:
            print(f"  - missing {gold} <- {q}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
