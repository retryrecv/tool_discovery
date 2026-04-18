"""Daily rebuild orchestrator — the Phase 3 entry point.

What it does, in order, per customer:

  1. Hash the catalog.  Skip the rebuild entirely if the hash matches
     the last build AND the feedback delta is below threshold.
  2. Build the candidate Tree using `build_tree_index` with the current
     active config.  (Autotuning is wired in but defaults to off — it
     multiplies LLM cost by `len(grid)` so it should be opt-in until
     enrichment becomes async in Phase 4.)
  3. Score the candidate against the customer's golden samples
     combined with feedback-derived eval queries.
  4. Run the Phase 1 promotion gate.
  5. If a previous active version exists, write a `diff.json` next to
     the new snapshot summarising structural changes.

Returns a `RebuildOutcome` describing what happened.  Exit-code mapping
in the script entrypoint:
  0 = promoted
  1 = error
  2 = built but rejected by quality gate
  3 = skipped (no catalog change, no feedback delta)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..config import Config
from ..pipeline.orchestrator import build_tree_index
from ..router.layout import CustomerLayout
from ..router.promotion import PromotionResult, promote_if_better
from ..router.quality import QualityScore, compute_quality_score
from ..schema import RawTool, Tree
from ..storage import load_snapshot
from .catalog_hash import compute_catalog_hash, read_catalog_hash, write_catalog_hash
from .diff import SnapshotDiff, diff_snapshots
from .eval_adapter import EvalQuery, build_eval_set


@dataclass
class RebuildOutcome:
    customer_id: str
    skipped: bool
    reason: str
    candidate_version: str | None = None
    promotion: PromotionResult | None = None
    quality: QualityScore | None = None
    diff: SnapshotDiff | None = None
    feedback_eval_size: int = 0

    def to_dict(self) -> dict:
        return {
            "customer_id": self.customer_id,
            "skipped": self.skipped,
            "reason": self.reason,
            "candidate_version": self.candidate_version,
            "promotion": self.promotion.to_dict() if self.promotion else None,
            "quality": self.quality.to_dict() if self.quality else None,
            "diff": self.diff.summary() if self.diff else None,
            "feedback_eval_size": self.feedback_eval_size,
        }


def _count_feedback_rows(layout: CustomerLayout) -> int:
    fb_dir = layout.root / "feedback"
    if not fb_dir.exists():
        return 0
    n = 0
    for f in fb_dir.glob("*.jsonl"):
        n += sum(1 for line in f.read_text().splitlines() if line.strip())
    return n


def _last_seen_feedback_count(layout: CustomerLayout) -> int:
    p = layout.root / "feedback_seen.count"
    if not p.exists():
        return 0
    try:
        return int(p.read_text().strip())
    except ValueError:
        return 0


def _record_feedback_count(layout: CustomerLayout, n: int) -> None:
    (layout.root / "feedback_seen.count").write_text(str(n))


def rebuild_customer(
    customer_id: str,
    raw_tools: list[RawTool],
    config: Config,
    *,
    snapshots_root: str | Path,
    out_root: str | Path | None = None,
    feedback_delta_threshold: int = 50,
    epsilon: float = 0.02,
    force: bool = False,
    extra_score_fn: Callable[[Tree], dict] | None = None,
) -> RebuildOutcome:
    layout = CustomerLayout.for_customer(snapshots_root, customer_id)
    layout.ensure()
    out_root = Path(out_root) if out_root else layout.root

    new_hash = compute_catalog_hash(raw_tools)
    old_hash = read_catalog_hash(snapshots_root, customer_id)
    fb_now = _count_feedback_rows(layout)
    fb_seen = _last_seen_feedback_count(layout)
    fb_delta = fb_now - fb_seen

    if not force and old_hash == new_hash and fb_delta < feedback_delta_threshold:
        return RebuildOutcome(
            customer_id=customer_id,
            skipped=True,
            reason=(
                f"catalog unchanged (hash={new_hash[:8]}) and feedback delta "
                f"{fb_delta} < {feedback_delta_threshold}"
            ),
        )

    tree = build_tree_index(raw_tools, config, out_root=out_root, strict=False)
    candidate_version = tree.version

    eval_set = build_eval_set(snapshots_root, customer_id)
    samples_path = layout.samples_path()
    quality = compute_quality_score(tree, samples_path, config.embedder, k=config.recall_k)

    if eval_set:
        from ..retrieval.traverser import retrieve as _retrieve
        hits = 0
        anti_hits = 0
        pos = 0
        neg = 0
        for q in eval_set:
            q_emb = config.embedder.embed(q.query)
            retrieved = set(_retrieve(tree, q_emb, k=config.recall_k))
            if q.polarity == "positive":
                pos += 1
                if q.tool_id in retrieved:
                    hits += 1
            else:
                neg += 1
                if q.tool_id in retrieved:
                    anti_hits += 1
        pos_rate = hits / pos if pos else 1.0
        neg_rate = anti_hits / neg if neg else 0.0
        blended = 0.5 * quality.score + 0.5 * (pos_rate - 0.5 * neg_rate)
        quality = QualityScore(
            score=max(0.0, min(1.0, blended)),
            sample_count=quality.sample_count + len(eval_set),
            hits=quality.hits + hits,
            k=quality.k,
            misses=quality.misses,
        )

    promotion = promote_if_better(layout, candidate_version, quality, epsilon=epsilon)

    diff = None
    if promotion.previous_version:
        try:
            diff = diff_snapshots(
                layout.version_dir(promotion.previous_version),
                layout.version_dir(candidate_version),
            )
            (layout.version_dir(candidate_version) / "diff.json").write_text(
                json.dumps(diff.to_dict(), indent=2)
            )
        except FileNotFoundError:
            pass

    write_catalog_hash(snapshots_root, customer_id, new_hash)
    _record_feedback_count(layout, fb_now)

    return RebuildOutcome(
        customer_id=customer_id,
        skipped=False,
        reason="rebuilt",
        candidate_version=candidate_version,
        promotion=promotion,
        quality=quality,
        diff=diff,
        feedback_eval_size=len(eval_set),
    )
