from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .models import PlanStep


@dataclass(frozen=True)
class CapabilitySpec:
    capability_id: str
    description: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroundedAction:
    step_id: str
    capability_id: str
    score: float
    rationale: str


class ActionGrounder(Protocol):
    def ground_step(self, step: PlanStep) -> GroundedAction | None:
        ...


class SymbolicSubplanner(Protocol):
    def solve(self, goal_text: str) -> tuple[str, ...]:
        ...


@dataclass
class RegistryActionGrounder:
    capabilities: tuple[CapabilitySpec, ...] = field(default_factory=tuple)

    def register(self, capability: CapabilitySpec) -> None:
        self.capabilities = (*self.capabilities, capability)

    def ground_step(self, step: PlanStep) -> GroundedAction | None:
        best: CapabilitySpec | None = None
        best_score = 0.0
        step_tokens = _tokenize(" ".join(filter(None, [step.title, step.instructions, step.capability_hint or ""])))
        for capability in self.capabilities:
            capability_tokens = _tokenize(
                " ".join([capability.capability_id, capability.description, *capability.tags])
            )
            score = len(step_tokens & capability_tokens) / max(len(step_tokens), 1)
            if step.capability_hint and step.capability_hint == capability.capability_id:
                score += 1.0
            if score > best_score:
                best = capability
                best_score = score
        if best is None or best_score <= 0:
            return None
        return GroundedAction(
            step_id=step.step_id,
            capability_id=best.capability_id,
            score=best_score,
            rationale=f"matched step content to {best.capability_id}",
        )


@dataclass
class MockSymbolicSubplanner:
    solutions: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def solve(self, goal_text: str) -> tuple[str, ...]:
        return self.solutions.get(goal_text, ())


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in "".join(char.lower() if char.isalnum() else " " for char in text).split()
        if token
    }
