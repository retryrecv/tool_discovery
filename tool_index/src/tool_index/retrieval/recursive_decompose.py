"""Experimental retrieval-aware recursive decomposition.

This module keeps the promoted one-shot decomposed retrieval path unchanged.
It adds a bounded ReAct/Self-Ask style loop for experiments: split, retrieve,
inspect retrieval confidence, then refine only weak steps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re

import numpy as np

from ..schema import ToolDescriptor, Tree
from .decompose import decompose_query
from .traverser import retrieve

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)
_ACTION_RE = re.compile(
    r"\b("
    r"add|calculate|compare|compress|convert|decode|detect|download|encode|extract|fetch|filter|format|"
    r"generate|get|hash|parse|pretty[- ]?print|pull|query|read|resize|scan|sort|summari[sz]e|"
    r"timestamp|transform|translate|validate|write"
    r")\b",
    re.IGNORECASE,
)
_SEQUENCE_RE = re.compile(r"\b(and|and then|then|after that|afterwards|before|finally|also|plus)\b|[,;]", re.IGNORECASE)

_PLAN_EXAMPLES = """Examples:
User query: Fetch this API URL, parse the URL first, pretty-print the JSON response, pull the id, and hash it.
JSON: [
  {"text": "parse the provided URL", "expected_tools": 1},
  {"text": "fetch the API URL", "expected_tools": 1},
  {"text": "pretty-print the JSON response", "expected_tools": 1},
  {"text": "pull the id from the JSON response", "expected_tools": 1},
  {"text": "hash the extracted id", "expected_tools": 1}
]

User query: Take this CSV of monthly sales, parse it, sort each branch by revenue, compute summary statistics, and calculate the percentage growth.
JSON: [
  {"text": "parse the sales CSV", "expected_tools": 1},
  {"text": "sort each branch by revenue", "expected_tools": 1},
  {"text": "compute summary statistics for monthly sales", "expected_tools": 1},
  {"text": "calculate percentage growth", "expected_tools": 1}
]

User query: Validate this ID scan, generate a case id, hash the case id, and timestamp the validation.
JSON: [
  {"text": "validate the ID scan image", "expected_tools": 1},
  {"text": "generate a case id", "expected_tools": 1},
  {"text": "hash the case id", "expected_tools": 1},
  {"text": "timestamp the validation", "expected_tools": 1}
]"""

_DEPENDENCY_HINT_EXAMPLES = """Examples:
User query: What weekday is today?
JSON: ["get the current date and time", "return the weekday for the current date"]

User query: Convert right now to the London timezone.
JSON: ["get the current date and time", "convert the current datetime to the London timezone"]

User query: Make a new tracking id, hash it, and timestamp the record.
JSON: ["generate a new UUID tracking id", "hash the tracking id", "get the current date and time for the timestamp"]"""


@dataclass(frozen=True)
class ScoredTool:
    """A reranked candidate with its MaxSim score."""

    tool_id: str
    score: float


@dataclass
class RecursiveStepTrace:
    """Trace for one leaf or refined planner step."""

    step_id: str
    text: str
    expected_tools: int
    circle: int
    parent_step_id: str | None
    decision: str
    reason: str
    candidates: list[str]
    ranked_tools: list[str]
    top_score: float | None
    margin: float | None
    refined_into: list[str] = field(default_factory=list)
    refined_expected_tools: list[int] = field(default_factory=list)


@dataclass
class RecursiveRetrieveResult:
    """Final tool set plus planner diagnostics."""

    tool_ids: list[str]
    traces: list[RecursiveStepTrace]
    unresolved_steps: list[RecursiveStepTrace]
    llm_calls: int


@dataclass(frozen=True)
class _Step:
    step_id: str
    text: str
    expected_tools: int
    circle: int
    parent_step_id: str | None = None


@dataclass(frozen=True)
class PlannedStep:
    """Structured LLM planner step."""

    text: str
    expected_tools: int = 1


def _cos(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    return float(va @ vb / (na * nb)) if na and nb else 0.0


def score_tools(
    query_embedding: list[float],
    candidates: list[str],
    tool_vectors: dict[str, list[list[float]]],
) -> list[ScoredTool]:
    """Return candidate tools ranked by MaxSim score."""
    scored: list[ScoredTool] = []
    for tid in candidates:
        vecs = tool_vectors.get(tid, [])
        score = max((_cos(query_embedding, v) for v in vecs), default=0.0)
        scored.append(ScoredTool(tool_id=tid, score=score))
    return sorted(scored, key=lambda item: -item.score)


def _strip_fence(text: str) -> str:
    m = _FENCE_RE.match(text)
    return m.group(1) if m else text


def _coerce_expected_tools(value) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 1
    return min(max(n, 1), 8)


def _parse_planned_steps(raw: str, fallback_text: str, max_sub_queries: int) -> list[PlannedStep]:
    """Parse structured planner output, accepting legacy string lists too."""
    try:
        parsed = json.loads(_strip_fence(raw).strip())
    except json.JSONDecodeError:
        return [PlannedStep(fallback_text, expected_tools=1)]

    if not isinstance(parsed, list) or not parsed:
        return [PlannedStep(fallback_text, expected_tools=1)]

    out: list[PlannedStep] = []
    for item in parsed:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            out.append(PlannedStep(text=text, expected_tools=_coerce_expected_tools(item.get("expected_tools", 1))))
        elif isinstance(item, (str, int, float)) and str(item).strip():
            out.append(PlannedStep(text=str(item).strip(), expected_tools=1))
        if len(out) >= max_sub_queries:
            break

    return out or [PlannedStep(fallback_text, expected_tools=1)]


def build_plan_prompt(query: str, schema_lines: list[str]) -> str:
    """Build the initial structured decomposition prompt."""
    schema_block = "\n".join(f"- {s}" for s in schema_lines) if schema_lines else "(no schema provided)"
    return (
        "You break a user query into tool-search steps.\n"
        "Each step must say whether it is already atomic or still covers multiple tool calls.\n\n"
        "Catalog capability areas:\n"
        f"{schema_block}\n\n"
        "Rules:\n"
        "- Output JSON only: a list of objects.\n"
        "- Object schema: {\"text\": string, \"expected_tools\": integer}.\n"
        "- Use expected_tools=1 only when the step should map to exactly one tool.\n"
        "- If a step still includes multiple operations, set expected_tools to that operation count or split it.\n"
        "- Do not invent operations the user did not request.\n\n"
        f"{_PLAN_EXAMPLES}\n\n"
        f"User query: {query}\n"
        "JSON:"
    )


def plan_query_steps(
    query: str,
    llm,
    schema_lines: list[str],
    *,
    max_sub_queries: int = 8,
) -> list[PlannedStep]:
    """Return structured decomposition steps for a query."""
    try:
        raw = llm.call(build_plan_prompt(query, schema_lines))
    except Exception:
        return [PlannedStep(query, expected_tools=1)]
    return _parse_planned_steps(raw, query, max_sub_queries)


def _parse_string_steps(raw: str, fallback_text: str, max_sub_queries: int) -> list[str]:
    try:
        parsed = json.loads(_strip_fence(raw).strip())
    except json.JSONDecodeError:
        return [fallback_text]

    if not isinstance(parsed, list) or not parsed:
        return [fallback_text]

    cleaned = [str(item).strip() for item in parsed if isinstance(item, (str, int, float)) and str(item).strip()]
    return cleaned[:max_sub_queries] or [fallback_text]


def build_dependency_hinted_prompt(query: str, schema_lines: list[str]) -> str:
    """Build an experimental decomposer prompt with implicit dependency hints."""
    schema_block = "\n".join(f"- {s}" for s in schema_lines) if schema_lines else "(no schema provided)"
    return (
        "You break a user query into atomic sub-queries, one per tool call.\n"
        "The tool catalog covers these capability areas:\n"
        f"{schema_block}\n\n"
        "Implicit dependency rules:\n"
        "- If the user asks about today, now, right now, current time, or a timestamp, include a step to get the current date and time.\n"
        "- If the user asks to create a new id, tracking id, case id, token, or correlation id, include a step to generate a new UUID.\n"
        "- Keep derived operations separate, such as weekday lookup, timezone conversion, hashing, encoding, or validation.\n\n"
        "Rules:\n"
        "- One sub-query per atomic operation the user wants or implies as an input dependency.\n"
        "- Phrase each sub-query as a short standalone request.\n"
        "- Do not invent unrelated operations.\n"
        "- Output JSON only: a list of strings.\n\n"
        f"{_DEPENDENCY_HINT_EXAMPLES}\n\n"
        f"User query: {query}\n"
        "JSON:"
    )


def decompose_query_with_dependency_hints(
    query: str,
    llm,
    schema_lines: list[str],
    *,
    max_sub_queries: int = 8,
) -> list[str]:
    """Experimental decomposer that expands implicit time/id dependencies."""
    try:
        raw = llm.call(build_dependency_hinted_prompt(query, schema_lines))
    except Exception:
        return [query]
    return _parse_string_steps(raw, query, max_sub_queries)


def _candidate_lines(
    ranked: list[ScoredTool],
    descriptors_by_id: dict[str, ToolDescriptor] | None,
    *,
    limit: int = 5,
) -> list[str]:
    lines: list[str] = []
    for item in ranked[:limit]:
        desc = descriptors_by_id.get(item.tool_id) if descriptors_by_id else None
        label = desc.name if desc else item.tool_id
        doc = desc.original_doc.strip().replace("\n", " ") if desc else ""
        if len(doc) > 120:
            doc = doc[:119].rstrip() + "..."
        suffix = f" - {doc}" if doc else ""
        lines.append(f"- {label} ({item.score:.3f}){suffix}")
    return lines


def build_refine_prompt(
    step_text: str,
    expected_tools: int,
    reason: str,
    ranked: list[ScoredTool],
    schema_lines: list[str],
    descriptors_by_id: dict[str, ToolDescriptor] | None = None,
) -> str:
    """Build the prompt for refining one weak planner step."""
    schema_block = "\n".join(f"- {s}" for s in schema_lines) if schema_lines else "(no schema provided)"
    candidates = "\n".join(_candidate_lines(ranked, descriptors_by_id)) or "(no candidates)"
    return (
        "You refine one tool-search sub-query.\n"
        "The goal is one atomic sub-query per actual tool call.\n"
        "If the sub-query hides multiple operations, split it. If it is just poorly phrased, rewrite it.\n"
        "If no supported operation is present, return an empty JSON list.\n\n"
        "Catalog capability areas:\n"
        f"{schema_block}\n\n"
        f"Current sub-query: {step_text}\n"
        f"Expected tool calls covered by this sub-query: {expected_tools}\n"
        f"Retrieval problem: {reason}\n"
        "Top candidate tools from the current search:\n"
        f"{candidates}\n\n"
        "Rules:\n"
        "- Output JSON only: a list of objects.\n"
        "- Object schema: {\"text\": string, \"expected_tools\": integer}.\n"
        "- Prefer expected_tools=1 leaves; only use a larger number if the step still cannot be split.\n"
        "- The sum of expected_tools in the output must preserve the current sub-query coverage.\n"
        "- Do not add operations not requested by the original sub-query.\n"
        "- Prefer concrete operation names over broad workflow language.\n\n"
        f"{_PLAN_EXAMPLES}\n\n"
        "JSON:"
    )


def _decision(
    step_text: str,
    expected_tools: int,
    ranked: list[ScoredTool],
    *,
    accept_score: float,
    accept_margin: float,
) -> tuple[str, str, float | None, float | None]:
    if expected_tools != 1:
        return "refine", "expected_multi_tool_step", None, None
    if not ranked:
        return "unresolved", "no_candidates", None, None
    top = ranked[0].score
    margin = top - ranked[1].score if len(ranked) > 1 else None
    if top < accept_score:
        return "refine", "low_score", top, margin
    if margin is not None and margin < accept_margin:
        return "refine", "low_margin", top, margin
    if _looks_multi_intent(step_text):
        return "refine", "multi_intent_text", top, margin
    return "resolved", "accepted", top, margin


def _looks_multi_intent(text: str) -> bool:
    """Cheap guard for confident but still broad workflow steps."""
    lowered = text.lower()
    if not _SEQUENCE_RE.search(lowered):
        return False
    actions = {match.group(0).lower() for match in _ACTION_RE.finditer(lowered)}
    return len(actions) >= 2


def _dedupe_refinements(children: list[PlannedStep], original: str, max_sub_queries: int) -> list[PlannedStep]:
    seen: set[str] = set()
    out: list[PlannedStep] = []
    original_norm = original.strip().lower()
    for child in children:
        norm = child.text.strip().lower()
        if not norm or norm == original_norm or norm in seen:
            continue
        seen.add(norm)
        out.append(PlannedStep(text=child.text.strip(), expected_tools=child.expected_tools))
        if len(out) >= max_sub_queries:
            break
    return out


def _coverage_preserved(parent_expected_tools: int, children: list[PlannedStep]) -> bool:
    return sum(child.expected_tools for child in children) >= parent_expected_tools


def _pool_ranked(pool: dict[str, float], ranked: list[ScoredTool], *, k: int) -> None:
    """Pool ranked tools by best rank, matching retrieve_decomposed scoring."""
    for rank, item in enumerate(ranked[:k]):
        score = float(k - rank)
        if score > pool.get(item.tool_id, 0.0):
            pool[item.tool_id] = score


def retrieve_refined_decomposed(
    tree: Tree,
    query: str,
    llm,
    embedder,
    tool_vectors: dict[str, list[list[float]]],
    schema_lines: list[str],
    *,
    k: int,
    rerank_k: int,
    beam: int,
    max_circles: int = 2,
    max_sub_queries: int = 8,
    max_refinements: int = 20,
    accept_score: float = 0.35,
    accept_margin: float = 0.03,
    descriptors_by_id: dict[str, ToolDescriptor] | None = None,
) -> RecursiveRetrieveResult:
    """Baseline decomposition plus bounded refinement for weak sub-queries.

    The first circle uses the promoted ``decompose_query`` planner. Every
    original sub-query is still searched and pooled, so this experiment tests
    whether recursive refinements can add missing tools without replacing the
    baseline split strategy.
    """
    max_circles = max(1, max_circles)
    initial = decompose_query(query, llm, schema_lines, max_sub_queries=max_sub_queries)
    llm_calls = 1
    queue: list[_Step] = [
        _Step(step_id=f"s{i + 1}", text=text, expected_tools=1, circle=0)
        for i, text in enumerate(initial)
    ]
    traces: list[RecursiveStepTrace] = []
    unresolved: list[RecursiveStepTrace] = []
    pool: dict[str, float] = {}
    next_id = len(queue) + 1
    refinements_used = 0

    while queue:
        step = queue.pop(0)
        q_emb = embedder.embed(step.text)
        candidates = retrieve(tree, q_emb, k=rerank_k, beam=beam)
        ranked = score_tools(q_emb, candidates, tool_vectors)
        _pool_ranked(pool, ranked, k=k)
        decision, reason, top_score, margin = _decision(
            step.text,
            step.expected_tools,
            ranked,
            accept_score=accept_score,
            accept_margin=accept_margin,
        )
        can_refine = (
            decision == "refine"
            and step.circle + 1 < max_circles
            and refinements_used < max_refinements
        )
        trace = RecursiveStepTrace(
            step_id=step.step_id,
            text=step.text,
            expected_tools=step.expected_tools,
            circle=step.circle,
            parent_step_id=step.parent_step_id,
            decision=decision,
            reason=reason,
            candidates=list(candidates),
            ranked_tools=[item.tool_id for item in ranked[:k]],
            top_score=top_score,
            margin=margin,
        )

        if decision == "resolved":
            traces.append(trace)
            continue

        if can_refine:
            refinements_used += 1
            prompt = build_refine_prompt(step.text, step.expected_tools, reason, ranked, schema_lines, descriptors_by_id)
            try:
                raw = llm.call(prompt)
            except Exception:
                raw = "[]"
            llm_calls += 1
            children = _dedupe_refinements(
                _parse_planned_steps(raw, step.text, max_sub_queries),
                step.text,
                max_sub_queries,
            )
            if children and not _coverage_preserved(step.expected_tools, children):
                children = []
                trace.reason = "refine_lost_coverage"
            trace.refined_into = [child.text for child in children]
            trace.refined_expected_tools = [child.expected_tools for child in children]
            traces.append(trace)
            if children:
                for child in children:
                    child_id = f"s{next_id}"
                    next_id += 1
                    queue.append(
                        _Step(
                            step_id=child_id,
                            text=child.text,
                            expected_tools=child.expected_tools,
                            circle=step.circle + 1,
                            parent_step_id=step.step_id,
                        )
                    )
                continue
            trace.decision = "unresolved"
            trace.reason = "refine_empty"
            unresolved.append(trace)
            continue

        trace.decision = "unresolved"
        if decision == "refine":
            suffix = "max_refinements" if refinements_used >= max_refinements else "max_circles"
            trace.reason = f"{reason}_{suffix}"
        traces.append(trace)
        unresolved.append(trace)

    ordered = [tid for tid, _ in sorted(pool.items(), key=lambda item: -item[1])[:k]]
    return RecursiveRetrieveResult(
        tool_ids=ordered,
        traces=traces,
        unresolved_steps=unresolved,
        llm_calls=llm_calls,
    )


def retrieve_dependency_hinted_decomposed(
    tree: Tree,
    query: str,
    llm,
    embedder,
    tool_vectors: dict[str, list[list[float]]],
    schema_lines: list[str],
    *,
    k: int,
    rerank_k: int,
    beam: int,
    max_sub_queries: int = 8,
) -> RecursiveRetrieveResult:
    """Run dependency-hinted decomposition through ordinary union retrieval."""
    sub_queries = decompose_query_with_dependency_hints(
        query,
        llm,
        schema_lines,
        max_sub_queries=max_sub_queries,
    )
    traces: list[RecursiveStepTrace] = []
    pool: dict[str, float] = {}
    for i, sub_query in enumerate(sub_queries):
        q_emb = embedder.embed(sub_query)
        candidates = retrieve(tree, q_emb, k=rerank_k, beam=beam)
        ranked = score_tools(q_emb, candidates, tool_vectors)
        _pool_ranked(pool, ranked, k=k)
        top_score = ranked[0].score if ranked else None
        margin = ranked[0].score - ranked[1].score if len(ranked) > 1 else None
        traces.append(
            RecursiveStepTrace(
                step_id=f"s{i + 1}",
                text=sub_query,
                expected_tools=1,
                circle=0,
                parent_step_id=None,
                decision="resolved" if ranked else "unresolved",
                reason="accepted" if ranked else "no_candidates",
                candidates=list(candidates),
                ranked_tools=[item.tool_id for item in ranked[:k]],
                top_score=top_score,
                margin=margin,
            )
        )
    ordered = [tid for tid, _ in sorted(pool.items(), key=lambda item: -item[1])[:k]]
    unresolved = [trace for trace in traces if trace.decision == "unresolved"]
    return RecursiveRetrieveResult(
        tool_ids=ordered,
        traces=traces,
        unresolved_steps=unresolved,
        llm_calls=1,
    )


def retrieve_recursive_decomposed(
    tree: Tree,
    query: str,
    llm,
    embedder,
    tool_vectors: dict[str, list[list[float]]],
    schema_lines: list[str],
    *,
    k: int,
    rerank_k: int,
    beam: int,
    max_circles: int = 3,
    max_sub_queries: int = 8,
    max_refinements: int = 20,
    accept_score: float = 0.35,
    accept_margin: float = 0.03,
    descriptors_by_id: dict[str, ToolDescriptor] | None = None,
) -> RecursiveRetrieveResult:
    """Resolve a query through bounded recursive split/retrieve/refine loops.

    ``max_circles`` counts total planner circles. ``3`` allows initial steps
    at circle 0 plus refinements at circles 1 and 2.
    """
    max_circles = max(1, max_circles)
    initial = plan_query_steps(query, llm, schema_lines, max_sub_queries=max_sub_queries)
    llm_calls = 1
    queue: list[_Step] = [
        _Step(step_id=f"s{i + 1}", text=step.text, expected_tools=step.expected_tools, circle=0)
        for i, step in enumerate(initial)
    ]
    traces: list[RecursiveStepTrace] = []
    unresolved: list[RecursiveStepTrace] = []
    pool: dict[str, float] = {}
    next_id = len(queue) + 1
    refinements_used = 0

    while queue:
        step = queue.pop(0)
        q_emb = embedder.embed(step.text)
        candidates = retrieve(tree, q_emb, k=rerank_k, beam=beam)
        ranked = score_tools(q_emb, candidates, tool_vectors)
        decision, reason, top_score, margin = _decision(
            step.text,
            step.expected_tools,
            ranked,
            accept_score=accept_score,
            accept_margin=accept_margin,
        )

        can_refine = (
            decision == "refine"
            and step.circle + 1 < max_circles
            and refinements_used < max_refinements
        )
        trace = RecursiveStepTrace(
            step_id=step.step_id,
            text=step.text,
            expected_tools=step.expected_tools,
            circle=step.circle,
            parent_step_id=step.parent_step_id,
            decision=decision,
            reason=reason,
            candidates=list(candidates),
            ranked_tools=[item.tool_id for item in ranked[:k]],
            top_score=top_score,
            margin=margin,
        )

        if decision == "resolved":
            item = ranked[0]
            pool[item.tool_id] = max(pool.get(item.tool_id, float("-inf")), item.score)
            traces.append(trace)
            continue

        if can_refine:
            refinements_used += 1
            prompt = build_refine_prompt(step.text, step.expected_tools, reason, ranked, schema_lines, descriptors_by_id)
            try:
                raw = llm.call(prompt)
            except Exception:
                raw = "[]"
            llm_calls += 1
            children = _dedupe_refinements(
                _parse_planned_steps(raw, step.text, max_sub_queries),
                step.text,
                max_sub_queries,
            )
            if children and not _coverage_preserved(step.expected_tools, children):
                children = []
                trace.reason = "refine_lost_coverage"
            trace.refined_into = [child.text for child in children]
            trace.refined_expected_tools = [child.expected_tools for child in children]
            traces.append(trace)
            if children:
                for child in children:
                    child_id = f"s{next_id}"
                    next_id += 1
                    queue.append(
                        _Step(
                            step_id=child_id,
                            text=child.text,
                            expected_tools=child.expected_tools,
                            circle=step.circle + 1,
                            parent_step_id=step.step_id,
                        )
                    )
                continue
            trace.decision = "unresolved"
            trace.reason = "refine_empty"
            unresolved.append(trace)
            continue

        if decision == "refine":
            trace.decision = "unresolved"
            suffix = "max_refinements" if refinements_used >= max_refinements else "max_circles"
            trace.reason = f"{reason}_{suffix}"
        traces.append(trace)
        unresolved.append(trace)

    ordered = [tid for tid, _ in sorted(pool.items(), key=lambda item: -item[1])[:k]]
    return RecursiveRetrieveResult(
        tool_ids=ordered,
        traces=traces,
        unresolved_steps=unresolved,
        llm_calls=llm_calls,
    )
