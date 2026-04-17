"""Heuristic fake LLM — the `LLMProvider` used in tests and offline builds.

Implements just enough of the prompt-response shapes the pipeline sends
to produce plausible structured outputs without any actual model. Four
prompt kinds are recognized (routed by `schema` kwarg or prompt marker):

    enrich_tool             → JSON enrichment
    describe_cluster        → one-line label
    judge_discriminability  → float in [0, 1]
    synthesize_queries      → JSON list of query strings

Deterministic: same prompt in, same response out. That's what keeps
golden tests stable.
"""
from __future__ import annotations
import json
import re


class FakeLLMProvider:
    """Template-driven mock LLM for offline runs."""

    def __init__(self, model_id: str = "fake-llm-v1"):
        """``model_id`` is used in cache keys and build traces — default
        value makes tests' cache paths stable across runs."""
        self.id = model_id

    def call(self, prompt: str, *, schema: str = "") -> str:
        """Dispatch to the appropriate heuristic based on ``schema``.

        Also sniffs for ALL_CAPS markers in the prompt as a fallback
        when the caller didn't set ``schema`` — useful for ad-hoc tests.
        Unknown prompts return empty string, which propagates to the
        pipeline's JSON fallbacks.
        """
        if schema == "enrich_tool" or "ENRICH_TOOL" in prompt:
            return self._enrich(prompt)
        if schema == "describe_cluster" or "DESCRIBE_CLUSTER" in prompt:
            return self._describe(prompt)
        if schema == "judge_discriminability" or "JUDGE_DISCRIMINABILITY" in prompt:
            return self._judge(prompt)
        if schema == "synthesize_queries" or "SYNTHESIZE_QUERIES" in prompt:
            return self._synth_queries(prompt)
        return ""

    # ------------------------------------------------------------------ enrich
    def _enrich(self, prompt: str) -> str:
        """Produce a plausible `Enrichment` from NAME + DOC fields in the prompt.

        Verb/noun inference is keyword-based — good enough to get
        similar tools pointing at the same embedding region.
        """
        name = _extract(prompt, r"NAME:\s*(.+)") or ""
        doc = _extract(prompt, r"DOC:\s*(.+)") or ""
        tokens = _keywords(f"{name} {doc}")
        verb = _verb_from_name(name)
        noun = _noun_from_tokens(tokens) or "record"
        intent = f"{verb} a {noun}"
        # Include verb + a sibling verb + a generic action verb; dedupe.
        synonyms = list({verb, _sibling_verb(verb), "manage"})
        queries = [
            f"{verb} the {noun}",
            f"please {verb} a {noun} for me",
            f"how do I {verb} a {noun}",
            f"{synonyms[0]} {noun} in the system",
            f"can you {verb} this {noun}",
        ]
        return json.dumps({
            "intent_phrase": intent,
            "input_kind": f"{noun} specification",
            "output_kind": f"resulting {noun}",
            "synonyms": synonyms,
            "example_queries": queries,
        })

    # ---------------------------------------------------------------- describe
    def _describe(self, prompt: str) -> str:
        """Derive a cluster label from distinctive tokens.

        Picks the first few tokens from MEMBERS that don't appear in
        NEIGHBORS — an extremely simple stand-in for "contrastive"
        description, but enough to keep sibling labels distinguishable
        in tests.
        """
        members = _extract_block(prompt, "MEMBERS:")
        neighbors = _extract_block(prompt, "NEIGHBORS:")
        mtoks = _keywords(members)
        ntoks = _keywords(neighbors)
        uniq = [t for t in mtoks if t not in set(ntoks)][:3]
        if not uniq:
            uniq = mtoks[:3]
        head = ", ".join(uniq) if uniq else "operations"
        return f"Tools focused on {head}"

    # ------------------------------------------------------------------- judge
    def _judge(self, prompt: str) -> str:
        """Score A vs B on Jaccard dissimilarity of their keyword sets.

        Low token overlap → high discriminability. Returns a float in
        ``[0, 1]`` formatted as a string (matching the real LLM contract).
        """
        a = _extract(prompt, r"A:\s*(.+)") or ""
        b = _extract(prompt, r"B:\s*(.+)") or ""
        ta, tb = set(_keywords(a)), set(_keywords(b))
        if not ta and not tb:
            return "0.5"
        jacc = len(ta & tb) / max(1, len(ta | tb))
        # 1 - Jaccard ⇒ higher is more distinct, matching the real prompt's scale.
        return f"{1.0 - jacc:.3f}"

    # --------------------------------------------------------- synth queries
    def _synth_queries(self, prompt: str) -> str:
        """Generate ``N`` eval queries by templating the tool's INTENT.

        Returns a JSON list of at most ``n`` strings — matches the real
        prompt's expected format so stage 5 can parse it identically.
        """
        intent = _extract(prompt, r"INTENT:\s*(.+)") or "do something"
        n = int(_extract(prompt, r"N:\s*(\d+)") or 5)
        verb, rest = _split_intent(intent)
        base = [
            f"please {intent}",
            f"I need to {intent}",
            f"help me {intent}",
            f"how can I {intent}",
            f"{verb} {rest} now",
            f"{verb} the {rest} quickly",
            f"{intent} for my project",
        ]
        return json.dumps(base[:n])


# -------------------------------------------------------- string helpers


def _extract(prompt: str, pattern: str) -> str | None:
    """Return the first regex capture group, stripped, or None."""
    m = re.search(pattern, prompt)
    return m.group(1).strip() if m else None


def _extract_block(prompt: str, header: str) -> str:
    """Return the block following ``header`` up to the next ALL_CAPS header.

    Used to pull MEMBERS / NEIGHBORS sections out of the describe_cluster
    prompt. The prompt structure is stable (we control both ends), so
    the simple regex-based parser is fine.
    """
    if header not in prompt:
        return ""
    tail = prompt.split(header, 1)[1]
    # End-of-block marker: a newline followed by ≥3 uppercase letters and a colon.
    m = re.search(r"\n[A-Z_]{3,}:\n", tail)
    return tail[:m.start()] if m else tail


# English-ish stop words + tool-boilerplate noise. Keeping this short and
# hand-curated — we'd rather miss a few signal tokens than include noise.
_STOP = {
    "the", "a", "an", "of", "to", "in", "for", "on", "and", "or", "by",
    "with", "from", "as", "is", "at", "be", "this", "that", "it", "its",
    "tool", "function", "method", "api", "given", "returns", "return",
    "into", "using", "use", "will", "can", "does", "do", "if", "not",
}


def _keywords(text: str) -> list[str]:
    """Extract distinctive keywords (no stop words, no dupes, min length 3).

    Order-preserving dedupe so the first occurrence wins — keeps
    description generation deterministic.
    """
    toks = re.findall(r"[a-zA-Z][a-zA-Z_0-9]+", text.lower())
    seen: set[str] = set()
    out: list[str] = []
    for t in toks:
        if t in _STOP or len(t) < 3:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


# Common action verbs. Used for verb inference from tool names and for
# the synonym mapping in `_sibling_verb`. Not exhaustive — additions are
# cheap if tests surface a missing one.
_VERBS = {
    "get", "list", "read", "fetch", "search", "find", "query", "scan",
    "create", "insert", "add", "write", "update", "set", "put", "upsert",
    "delete", "remove", "drop", "truncate",
    "send", "post", "publish", "notify",
    "download", "upload", "open", "close", "move", "copy", "rename",
    "parse", "compute", "calculate", "transform", "convert", "render",
}


def _verb_from_name(name: str) -> str:
    """Pick a verb from a tool name like ``db_users_read`` or ``readUser``.

    Tries split-on-separators first, then prefix match; falls back to
    ``"perform"`` so callers always get a verb.
    """
    for part in re.split(r"[_\-\.\s]+", name.lower()):
        if part in _VERBS:
            return part
    # Fall back to prefix scan — catches camelCase-ish names.
    for v in _VERBS:
        if name.lower().startswith(v):
            return v
    return "perform"


def _noun_from_tokens(tokens: list[str]) -> str:
    """Return the first non-verb token as the "noun" of the intent phrase."""
    for t in tokens:
        if t not in _VERBS:
            return t
    return ""


# Map each verb to a synonymous one, for synonym-list diversity.
# Asymmetric deliberately — we want a single stable mapping, not a
# generated equivalence class.
_VERB_SIBLINGS = {
    "get": "read", "list": "read", "fetch": "read", "read": "fetch",
    "create": "insert", "insert": "create", "add": "create",
    "update": "set", "set": "update",
    "delete": "remove", "remove": "delete",
    "send": "post", "post": "send",
}


def _sibling_verb(v: str) -> str:
    """Return a verb synonymous with ``v``, or ``v`` itself if none known."""
    return _VERB_SIBLINGS.get(v, v)


def _split_intent(intent: str) -> tuple[str, str]:
    """Split an intent phrase into ``(verb, remainder)``.

    Used to interpolate verb + remainder separately in synthesized
    queries. Single-word intents return ``(intent, "")``.
    """
    parts = intent.strip().split(" ", 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]
