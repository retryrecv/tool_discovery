"""`RawTool` — the input shape passed to `build_tree_index` and the stage scripts.

Mirrors the ``ToolDefinition`` interface from the TypeScript catalogs in
``data/rawTools/*.tools.ts``, flattened to the fields the Python pipeline
actually needs. Stage 1 (`pipeline/stage1_normalize.py`) converts each
`RawTool` into a `ToolDescriptor`; later stages never touch the raw dict.

Lives in `schema/` (not `pipeline/`) so the generated catalog
(`data/generateTools/tools.py`) can import it without pulling in the
orchestrator.
"""
from __future__ import annotations
from typing import NotRequired, TypedDict


class RawTool(TypedDict):
    """One entry in the ``raw_tools`` list.

    Required fields
    ---------------
    name : str
        Unique tool identifier. Must be stable across catalog versions
        because it is used to derive the content-addressed
        ``ToolDescriptor.id`` (``hash(name + ":" + signature)``).

        Examples::

            "calculator"
            "science__al_quran__get_range_of_verses"

    signature : str
        Human-readable call signature summarising inputs and output,
        e.g. ``"calculator(expression) -> str"``. Used by stage-2
        enrichment prompts to ground the LLM's understanding of I/O shapes.

    doc : str
        Natural-language description of what the tool does. Sent verbatim
        to the LLM during enrichment; longer, more precise descriptions
        improve clustering quality.

    Optional fields
    ---------------
    id : str, optional
        Explicit stable identifier. When omitted, stage 1 derives one as
        ``new_id("tool", f"{name}:{signature}")``.

    source : str, optional
        Free-form provenance string (catalog filename, API spec URL).
        Informational only; never parsed by the pipeline.

    examples : list[dict], optional
        Worked call examples, each a dict with at least an ``"args"`` key
        and optionally a ``"returns"`` key. Fed into stage-2 enrichment
        prompts when present.

        Example::

            [{"args": {"expression": "(3 + 4) * 2"}, "returns": "14"}]
    """
    name: str
    signature: str
    doc: str
    id:       NotRequired[str]
    source:   NotRequired[str]
    examples: NotRequired[list[dict]]
