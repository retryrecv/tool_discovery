"""Evaluate recall@k against the hand-written natural-language test cases.

Reads SIMPLE_CASES + COMPLEX_CASES + ULTRA_COMPLEX_CASES from
data/corpus/eval_queries.py.
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
from collections import Counter
import importlib.util
import json
import sys
from pathlib import Path

from _pipeline_config import make_config, run_dir
from tool_index.retrieval import (
    decompose_query,
    inject_pattern_sub_queries,
    precompute_tool_vectors,
    retrieve_dependency_hinted_decomposed,
    retrieve_refined_decomposed,
    retrieve_recursive_decomposed,
)
from tool_index.retrieval.recursive_decompose import score_tools
from tool_index.retrieval.traverser import retrieve_with_path
from tool_index.schema import Enrichment, Tree, ToolDescriptor


def _load_test_cases(repo_root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    # catalog is already loaded by _pipeline_config under the
    # "corpus" package alias; reuse it so the relative import in
    # eval_queries.py resolves.
    spec = importlib.util.spec_from_file_location(
        "corpus.eval_queries", repo_root / "data/corpus/eval_queries.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["corpus.eval_queries"] = mod
    spec.loader.exec_module(mod)
    return mod.SIMPLE_CASES, mod.COMPLEX_CASES, getattr(mod, "ULTRA_COMPLEX_CASES", [])


def _summarize_node(desc: str, max_chars: int = 140) -> str:
    """One-line summary of a node description for prompt injection."""
    s = desc.strip().replace("\n", " ")
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "..."


def _schema_lines(tree: Tree) -> list[str]:
    """Return one-line summaries for the most informative schema level.

    Uses L1 if there is more than one L1 domain; otherwise falls back to
    L2 (categories) which carries the actual capability split at small
    catalog scales where everything compacts into a single L1.
    """
    l1 = [tree.nodes_by_id[cid] for cid in tree.root.children if cid in tree.nodes_by_id]
    chosen = l1 if len(l1) > 1 else [
        tree.nodes_by_id[ccid]
        for n in l1
        for ccid in n.children
        if ccid in tree.nodes_by_id
    ]
    return [_summarize_node(n.description) for n in chosen]



def _tool_name(tool_id: str, descriptors_by_id: dict[str, ToolDescriptor]) -> str:
    desc = descriptors_by_id.get(tool_id)
    return desc.name if desc else tool_id


def _tool_payload(tool_id: str, descriptors_by_id: dict[str, ToolDescriptor]) -> dict:
    desc = descriptors_by_id.get(tool_id)
    return {
        "tool_id": tool_id,
        "tool_name": desc.name if desc else tool_id,
    }


def _path_payload(tree: Tree, path: list[str]) -> list[dict]:
    out: list[dict] = []
    for node_id in path:
        node = tree.nodes_by_id.get(node_id)
        if node is None:
            out.append({"node_id": node_id})
            continue
        out.append(
            {
                "node_id": node.id,
                "level": node.level,
                "description": _summarize_node(node.description, max_chars=220),
            }
        )
    return out


def _ranked_payload(ranked, descriptors_by_id: dict[str, ToolDescriptor], *, limit: int) -> list[dict]:
    out: list[dict] = []
    for rank, item in enumerate(ranked[:limit], start=1):
        payload = _tool_payload(item.tool_id, descriptors_by_id)
        payload.update({"rank": rank, "score": item.score})
        out.append(payload)
    return out


def _mode_name(args) -> str:
    if args.dependency_hinted_decompose:
        return "dependency_hinted_decompose"
    if args.refined_decompose:
        return "refined_decompose"
    if args.recursive_decompose:
        return "recursive_decompose"
    if args.decompose and args.pattern_inject:
        return "decompose_pattern_inject"
    if args.decompose:
        return "decompose"
    return "direct"


def _recursive_trace_payload(result, descriptors_by_id: dict[str, ToolDescriptor]) -> dict:
    return {
        "tool_ids": result.tool_ids,
        "tool_names": [_tool_name(tid, descriptors_by_id) for tid in result.tool_ids],
        "llm_calls": result.llm_calls,
        "unresolved_step_ids": [trace.step_id for trace in result.unresolved_steps],
        "steps": [
            {
                "step_id": trace.step_id,
                "parent_step_id": trace.parent_step_id,
                "circle": trace.circle,
                "text": trace.text,
                "expected_tools": trace.expected_tools,
                "decision": trace.decision,
                "reason": trace.reason,
                "candidate_tool_ids": trace.candidates,
                "candidate_tool_names": [_tool_name(tid, descriptors_by_id) for tid in trace.candidates],
                "ranked_tool_ids": trace.ranked_tools,
                "ranked_tool_names": [_tool_name(tid, descriptors_by_id) for tid in trace.ranked_tools],
                "top_score": trace.top_score,
                "margin": trace.margin,
                "refined_into": trace.refined_into,
                "refined_expected_tools": trace.refined_expected_tools,
            }
            for trace in result.traces
        ],
    }

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rerank-k", type=int, default=20)
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument(
        "--decompose",
        action="store_true",
        help="Enable Direction 4 LLM query decomposition + union retrieval.",
    )
    ap.add_argument(
        "--recursive-decompose",
        action="store_true",
        help="Enable experimental retrieval-aware recursive decomposition.",
    )
    ap.add_argument(
        "--refined-decompose",
        action="store_true",
        help="Enable experimental baseline decomposition plus weak-step refinement.",
    )
    ap.add_argument(
        "--dependency-hinted-decompose",
        action="store_true",
        help="Enable experimental decomposition with implicit time/id dependency hints.",
    )
    ap.add_argument(
        "--pattern-inject",
        action="store_true",
        help="Append deterministic glue-tool sub-queries after --decompose.",
    )
    ap.add_argument("--max-circles", type=int, default=3)
    ap.add_argument("--max-refinements", type=int, default=20)
    ap.add_argument("--accept-score", type=float, default=0.35)
    ap.add_argument("--accept-margin", type=float, default=0.03)
    ap.add_argument(
        "--limit-cases",
        type=int,
        default=0,
        help="Limit each difficulty bucket to the first N cases for fast experiments.",
    )
    ap.add_argument(
        "--trace-out",
        type=Path,
        default=None,
        help="Write one JSONL trace record per evaluated case.",
    )
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

    schema_lines = (
        _schema_lines(tree)
        if args.decompose or args.recursive_decompose or args.refined_decompose or args.dependency_hinted_decompose
        else []
    )
    if args.decompose:
        print(f"Decomposition ON: injecting {len(schema_lines)} schema lines into LLM prompt")
        if args.pattern_inject:
            print("Pattern injector ON: appending deterministic glue-tool sub-queries")
    if args.recursive_decompose:
        print(
            "Recursive decomposition ON: "
            f"schema lines={len(schema_lines)}, max_circles={args.max_circles}, "
            f"max_refinements={args.max_refinements}, "
            f"accept_score={args.accept_score:.3f}, accept_margin={args.accept_margin:.3f}"
        )
    if args.refined_decompose:
        print(
            "Refined decomposition ON: "
            f"schema lines={len(schema_lines)}, max_circles={args.max_circles}, "
            f"max_refinements={args.max_refinements}, "
            f"accept_score={args.accept_score:.3f}, accept_margin={args.accept_margin:.3f}"
        )
    if args.dependency_hinted_decompose:
        print(f"Dependency-hinted decomposition ON: injecting {len(schema_lines)} schema lines into LLM prompt")

    simple_cases, complex_cases, ultra_complex_cases = _load_test_cases(repo_root)
    if args.limit_cases:
        simple_cases = simple_cases[: args.limit_cases]
        complex_cases = complex_cases[: args.limit_cases]
        ultra_complex_cases = ultra_complex_cases[: args.limit_cases]
    print(
        f"Test set: {len(simple_cases)} simple + {len(complex_cases)} complex + "
        f"{len(ultra_complex_cases)} ultra-complex = "
        f"{len(simple_cases)+len(complex_cases)+len(ultra_complex_cases)} cases"
    )

    descriptors_by_id = {d.id: d for d in descriptors}
    trace_records: list[dict] = []
    decompose_log: list[tuple[str, list[str]]] = []
    inject_log: list[tuple[str, list[str]]] = []
    recursive_log = []

    def trace_search(text: str, q_emb: list[float]) -> tuple[dict, list]:
        candidates, path = retrieve_with_path(tree, q_emb, k=args.rerank_k, beam=args.beam)
        ranked = score_tools(q_emb, candidates, tool_vectors)
        return (
            {
                "text": text,
                "route_path": _path_payload(tree, path),
                "candidate_tools": [_tool_payload(tid, descriptors_by_id) for tid in candidates],
                "reranked_tools": _ranked_payload(ranked, descriptors_by_id, limit=args.rerank_k),
                "top_k_tools": [_tool_payload(item.tool_id, descriptors_by_id) for item in ranked[: args.k]],
            },
            ranked,
        )

    def run_query(query: str) -> tuple[list[str], dict]:
        if args.dependency_hinted_decompose:
            result = retrieve_dependency_hinted_decomposed(
                tree,
                query,
                cfg.enricher_llm,
                cfg.embedder,
                tool_vectors,
                schema_lines,
                k=args.k,
                rerank_k=args.rerank_k,
                beam=args.beam,
            )
            recursive_log.append((query, result))
            return result.tool_ids, _recursive_trace_payload(result, descriptors_by_id)

        if args.refined_decompose:
            result = retrieve_refined_decomposed(
                tree,
                query,
                cfg.enricher_llm,
                cfg.embedder,
                tool_vectors,
                schema_lines,
                k=args.k,
                rerank_k=args.rerank_k,
                beam=args.beam,
                max_circles=args.max_circles,
                max_refinements=args.max_refinements,
                accept_score=args.accept_score,
                accept_margin=args.accept_margin,
                descriptors_by_id=descriptors_by_id,
            )
            recursive_log.append((query, result))
            return result.tool_ids, _recursive_trace_payload(result, descriptors_by_id)

        if args.recursive_decompose:
            result = retrieve_recursive_decomposed(
                tree,
                query,
                cfg.enricher_llm,
                cfg.embedder,
                tool_vectors,
                schema_lines,
                k=args.k,
                rerank_k=args.rerank_k,
                beam=args.beam,
                max_circles=args.max_circles,
                max_refinements=args.max_refinements,
                accept_score=args.accept_score,
                accept_margin=args.accept_margin,
                descriptors_by_id=descriptors_by_id,
            )
            recursive_log.append((query, result))
            return result.tool_ids, _recursive_trace_payload(result, descriptors_by_id)

        if not args.decompose:
            q_emb = cfg.embedder.embed(query)
            sub_trace, ranked = trace_search(query, q_emb)
            tool_ids = [item.tool_id for item in ranked[: args.k]]
            return tool_ids, {
                "mode": _mode_name(args),
                "sub_queries": [{"index": 0, "source": "original", **sub_trace}],
                "union_scores": [
                    {**_tool_payload(item.tool_id, descriptors_by_id), "score": float(args.k - rank)}
                    for rank, item in enumerate(ranked[: args.k])
                ],
            }

        llm_sub_queries = decompose_query(query, cfg.enricher_llm, schema_lines)
        final_sub_queries = list(llm_sub_queries)
        fired: list[str] = []
        if args.pattern_inject:
            final_sub_queries, fired = inject_pattern_sub_queries(query, final_sub_queries)
            inject_log.append((query, fired))
        decompose_log.append((query, final_sub_queries))

        llm_counts = Counter(sq.casefold().strip() for sq in llm_sub_queries)
        sub_traces: list[dict] = []
        pool: dict[str, float] = {}
        for index, sub_query in enumerate(final_sub_queries):
            source_key = sub_query.casefold().strip()
            if llm_counts[source_key] > 0:
                source = "llm"
                llm_counts[source_key] -= 1
            else:
                source = "injected"

            q_emb = cfg.embedder.embed(sub_query)
            sub_trace, ranked = trace_search(sub_query, q_emb)
            ranked_top_k = ranked[: args.k]
            for rank, item in enumerate(ranked_top_k):
                score = float(args.k - rank)
                if score > pool.get(item.tool_id, 0.0):
                    pool[item.tool_id] = score
            sub_traces.append({"index": index, "source": source, **sub_trace})

        union_ranked = sorted(pool.items(), key=lambda item: -item[1])[: args.k]
        tool_ids = [tid for tid, _ in union_ranked]
        return tool_ids, {
            "mode": _mode_name(args),
            "llm_sub_queries": llm_sub_queries,
            "injected_rules": fired,
            "final_sub_queries": final_sub_queries,
            "sub_queries": sub_traces,
            "union_scores": [
                {**_tool_payload(tid, descriptors_by_id), "score": score}
                for tid, score in union_ranked
            ],
        }

    def gold_resolution(gold_ids: list[str], topk: list[str], retrieval_trace: dict) -> list[dict]:
        sub_queries = retrieval_trace.get("sub_queries", [])
        steps = retrieval_trace.get("steps", [])
        candidate_sets = [
            {tool["tool_id"] for tool in sub.get("candidate_tools", [])}
            for sub in sub_queries
        ]
        ranked_sets = [
            {tool["tool_id"] for tool in sub.get("reranked_tools", [])}
            for sub in sub_queries
        ]
        recursive_candidate_sets = [set(step.get("candidate_tool_ids", [])) for step in steps]
        recursive_ranked_sets = [set(step.get("ranked_tool_ids", [])) for step in steps]

        out: list[dict] = []
        for gold_id in gold_ids:
            if gold_id in topk:
                reason = "hit"
            elif any(gold_id in ranked for ranked in ranked_sets + recursive_ranked_sets):
                reason = "union_miss"
            elif any(gold_id in candidates for candidates in candidate_sets + recursive_candidate_sets):
                reason = "rerank_miss"
            else:
                reason = "route_miss"
            out.append({**_tool_payload(gold_id, descriptors_by_id), "result": reason})
        return out

    def evaluate_cases(difficulty: str, cases: list[dict]) -> tuple[int, int, list[tuple[str, list[str]]], int]:
        found_total = 0
        total = 0
        misses: list[tuple[str, list[str]]] = []
        all_gold_cases = 0
        for index, case in enumerate(cases):
            gold_pairs = [(name, name_to_id.get(name)) for name in case["calls"]]
            gold_ids = [gid for _, gid in gold_pairs if gid is not None]
            if difficulty == "simple":
                simple_unmapped.extend(name for name, gid in gold_pairs if gid is None)
            if not gold_ids:
                continue

            topk, retrieval_trace = run_query(case["query"])
            found = [gid for gid in gold_ids if gid in topk]
            missing = [name for name, gid in gold_pairs if gid is not None and gid not in topk]
            total += len(gold_ids)
            found_total += len(found)
            if missing:
                misses.append((case["query"], missing))
            else:
                all_gold_cases += 1

            if args.trace_out:
                trace_records.append(
                    {
                        "case_id": f"{difficulty}-{index + 1:03d}",
                        "difficulty": difficulty,
                        "case_index": index,
                        "query": case["query"],
                        "mode": _mode_name(args),
                        "gold_tools": [
                            {"tool_name": name, "tool_id": gid}
                            for name, gid in gold_pairs
                        ],
                        "predicted_tools": [_tool_payload(tid, descriptors_by_id) for tid in topk],
                        "found_tools": [_tool_payload(tid, descriptors_by_id) for tid in found],
                        "missing_tools": [
                            {"tool_name": name, "tool_id": gid}
                            for name, gid in gold_pairs
                            if gid is not None and gid not in topk
                        ],
                        "gold_resolution": gold_resolution(gold_ids, topk, retrieval_trace),
                        "retrieval": retrieval_trace,
                    }
                )
        return found_total, total, misses, all_gold_cases

    simple_unmapped: list[str] = []
    simple_found, simple_seen, simple_miss_records, _simple_all_gold = evaluate_cases("simple", simple_cases)
    simple_hits = simple_found
    simple_misses = [(query, missing[0]) for query, missing in simple_miss_records if missing]

    complex_found, complex_total, complex_misses, complex_all_gold = evaluate_cases("complex", complex_cases)
    ultra_found, ultra_total, ultra_misses, ultra_all_gold = evaluate_cases("ultra-complex", ultra_complex_cases)

    print()
    print(f"SIMPLE recall@{args.k}: {simple_hits}/{simple_seen} = {simple_hits/simple_seen:.3f}  (unmapped tool names: {len(simple_unmapped)})")
    print(f"COMPLEX set-recall@{args.k}: {complex_found}/{complex_total} = {complex_found/complex_total:.3f}")
    print(f"COMPLEX cases with all gold present: {len(complex_cases)-len(complex_misses)}/{len(complex_cases)}")
    if ultra_complex_cases:
        print(f"ULTRA-COMPLEX set-recall@{args.k}: {ultra_found}/{ultra_total} = {ultra_found/ultra_total:.3f}")
        print(f"ULTRA-COMPLEX cases with all gold present: {len(ultra_complex_cases)-len(ultra_misses)}/{len(ultra_complex_cases)}")
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

    if ultra_misses:
        print(f"\nUltra-complex partial misses ({len(ultra_misses)}):")
        for q, missing in ultra_misses[:10]:
            print(f"  - missing {missing} <- {q[:80]}")

    if args.decompose and decompose_log:
        n_split = sum(1 for _, subs in decompose_log if len(subs) > 1)
        avg_subs = sum(len(subs) for _, subs in decompose_log) / len(decompose_log)
        print(f"\nDecomposition stats: {n_split}/{len(decompose_log)} queries split into >1 sub-query; avg sub-queries = {avg_subs:.2f}")
        print("Sample decompositions:")
        for q, subs in decompose_log[:5]:
            print(f"  - {q[:60]!r}")
            for s in subs:
                print(f"      -> {s!r}")
        if args.pattern_inject and inject_log:
            fired_queries = sum(1 for _, fired in inject_log if fired)
            total_fired = sum(len(fired) for _, fired in inject_log)
            print(
                "Pattern injector stats: "
                f"{fired_queries}/{len(inject_log)} queries matched; {total_fired} injected sub-queries"
            )

    if (args.recursive_decompose or args.refined_decompose or args.dependency_hinted_decompose) and recursive_log:
        total_steps = sum(len(result.traces) for _, result in recursive_log)
        unresolved_steps = sum(len(result.unresolved_steps) for _, result in recursive_log)
        llm_calls = sum(result.llm_calls for _, result in recursive_log)
        avg_circles = (
            sum((max((trace.circle for trace in result.traces), default=0) + 1) for _, result in recursive_log)
            / len(recursive_log)
        )
        reasons = Counter(trace.reason for _, result in recursive_log for trace in result.traces)
        decisions = Counter(trace.decision for _, result in recursive_log for trace in result.traces)
        print()
        print(
            "Recursive decomposition stats: "
            f"steps={total_steps}, unresolved={unresolved_steps}, "
            f"llm_calls={llm_calls}, avg circles/query={avg_circles:.2f}"
        )
        print(f"Recursive decisions: {dict(sorted(decisions.items()))}")
        print(f"Recursive reasons: {dict(sorted(reasons.items()))}")
        print("Sample recursive traces:")
        for q, result in recursive_log[:3]:
            print(f"  - {q[:60]!r}")
            for trace in result.traces[:6]:
                top = trace.ranked_tools[0] if trace.ranked_tools else "none"
                score = "n/a" if trace.top_score is None else f"{trace.top_score:.3f}"
                margin = "n/a" if trace.margin is None else f"{trace.margin:.3f}"
                print(
                    f"      [{trace.decision}/{trace.reason}] circle={trace.circle} "
                    f"expected={trace.expected_tools} top={top} score={score} margin={margin} <- {trace.text!r}"
                )
                for child, expected_tools in zip(trace.refined_into, trace.refined_expected_tools, strict=False):
                    print(f"          -> expected={expected_tools} {child!r}")

    if args.trace_out:
        args.trace_out.parent.mkdir(parents=True, exist_ok=True)
        with args.trace_out.open("w") as f:
            for record in trace_records:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"\nTrace written: {args.trace_out} ({len(trace_records)} records)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
