"""Auto-rollback gate — decide whether a freshly built snapshot is allowed
to become the active version for a customer.

Rule:
    promote if active is None (first ever snapshot)
    OR new.score >= active.score - epsilon

Epsilon defaults to 0.02 (2 percentage points of recall@k tolerance) so a
small noise dip doesn't pin us to an older tree forever. Tune per
customer if you have a strong sample set.

The gate writes the candidate's quality file regardless of outcome, so
operators can audit rejected snapshots.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .layout import CustomerLayout
from .quality import QualityScore


@dataclass
class PromotionResult:
    promoted: bool
    candidate_version: str
    candidate_score: float
    previous_version: str | None
    previous_score: float | None
    reason: str

    def to_dict(self) -> dict:
        return {
            "promoted": self.promoted,
            "candidate_version": self.candidate_version,
            "candidate_score": self.candidate_score,
            "previous_version": self.previous_version,
            "previous_score": self.previous_score,
            "reason": self.reason,
        }


def _read_score(layout: CustomerLayout, version: str | None) -> float | None:
    if version is None:
        return None
    p = layout.quality_path(version)
    if not p.exists():
        return None
    return float(json.loads(p.read_text()).get("score", 0.0))


def promote_if_better(
    layout: CustomerLayout,
    candidate_version: str,
    candidate_quality: QualityScore,
    *,
    epsilon: float = 0.02,
) -> PromotionResult:
    layout.ensure()
    layout.quality_path(candidate_version).write_text(
        json.dumps(candidate_quality.to_dict(), indent=2)
    )

    active = layout.read_active()
    prev_score = _read_score(layout, active)

    if active is None:
        layout.write_active(candidate_version)
        return PromotionResult(
            promoted=True,
            candidate_version=candidate_version,
            candidate_score=candidate_quality.score,
            previous_version=None,
            previous_score=None,
            reason="first snapshot for customer",
        )

    if prev_score is None:
        layout.write_active(candidate_version)
        return PromotionResult(
            promoted=True,
            candidate_version=candidate_version,
            candidate_score=candidate_quality.score,
            previous_version=active,
            previous_score=None,
            reason=f"previous version {active} has no recorded score",
        )

    if candidate_quality.score + 1e-9 >= prev_score - epsilon:
        layout.write_active(candidate_version)
        return PromotionResult(
            promoted=True,
            candidate_version=candidate_version,
            candidate_score=candidate_quality.score,
            previous_version=active,
            previous_score=prev_score,
            reason=f"score {candidate_quality.score:.3f} within tolerance of {prev_score:.3f} (eps={epsilon})",
        )

    return PromotionResult(
        promoted=False,
        candidate_version=candidate_version,
        candidate_score=candidate_quality.score,
        previous_version=active,
        previous_score=prev_score,
        reason=f"score {candidate_quality.score:.3f} below {prev_score:.3f} - {epsilon} (rolled back)",
    )
