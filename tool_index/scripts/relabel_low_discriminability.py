"""Re-label nodes flagged as low-discriminability.

Reads ``05_validation.json`` for warnings of the form
``low discriminability X vs Y: <score>``, gathers the offending node IDs
plus their siblings (so the new prompt has contrastive context), and
re-runs ``llm_describe_cluster`` on each. Writes back into the existing
``tree_draft.json`` (description + embedding only — children unchanged).

Usage:
    uv run scripts/relabel_low_discriminability.py --run raw-tools

The cache layer means each fresh prompt costs one new LLM call; if the
prompt for a node is unchanged, it's a cache hit (no spend).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from _pipeline_config import make_config, run_dir
from tool_index.labeling import llm_describe_cluster
from tool_index.schema import Tree


_WARN_RE = re.compile(r"low discriminability (\S+) vs (\S+):")


def _collect_targets(validation: dict) -> set[str]:
    targets: set[str] = set()
    for w in validation.get("warnings", []):
        m = _WARN_RE.match(w)
        if m:
            targets.add(m.group(1))
            targets.add(m.group(2))
    return targets


def _parent_of(tree: Tree) -> dict[str, str]:
    out: dict[str, str] = {}
    for nid, node in tree.nodes_by_id.items():
        for cid in node.children:
            out[cid] = nid
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()

    cfg = make_config()
    out = run_dir(args.run)

    validation = json.loads((out / "05_validation.json").read_text())
    targets = _collect_targets(validation)
    print(f"Targets to relabel: {len(targets)}")
    if not targets:
        print("Nothing to do.")
        return 0

    tree_payload = json.loads((out / "tree_draft.json").read_text())
    meta = tree_payload.pop("_meta", {})
    tree = Tree.from_dict(tree_payload)
    parent_of = _parent_of(tree)

    relabeled = 0
    for tid in sorted(targets):
        node = tree.nodes_by_id.get(tid)
        if node is None or node.level == "tool":
            continue
        # Members = each child's description.
        member_texts = [tree.nodes_by_id[c].description for c in node.children if c in tree.nodes_by_id]
        if not member_texts:
            continue
        # Neighbors = sibling nodes' children's descriptions (one block per sibling).
        parent_id = parent_of.get(tid)
        siblings: list[str] = []
        if parent_id and parent_id in tree.nodes_by_id:
            siblings = [s for s in tree.nodes_by_id[parent_id].children if s != tid and s in tree.nodes_by_id]
        neighbor_texts: list[list[str]] = []
        for sid in siblings[:3]:
            sib = tree.nodes_by_id[sid]
            sib_members = [tree.nodes_by_id[c].description for c in sib.children if c in tree.nodes_by_id]
            if sib_members:
                neighbor_texts.append(sib_members)

        new_desc = llm_describe_cluster(member_texts, neighbor_texts, cfg.labeler_llm, contrastive=True)
        if new_desc and new_desc != node.description:
            node.description = new_desc
            node.embedding = cfg.embedder.embed(new_desc)
            relabeled += 1
            print(f"  {tid}: {new_desc[:120]}...")

    print(f"Relabeled {relabeled} of {len(targets)} candidates")

    payload = tree.to_dict()
    payload["_meta"] = meta
    (out / "tree_draft.json").write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
