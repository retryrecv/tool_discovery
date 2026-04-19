from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping

from .models import EvalResult, PlanStep, PlanVersion, RunState

QUALITY_SCORE_WEIGHTS = {
    "task_success_rate": 0.30,
    "plan_validity_rate": 0.20,
    "state_transition_accuracy": 0.20,
    "grounding_coverage": 0.15,
    "replan_recovery_rate": 0.10,
    "search_efficiency": 0.05,
}


@dataclass
class PlanEvaluator:
    def evaluate(self, plan: PlanVersion, state: RunState) -> EvalResult:
        step_map = plan.step_map()
        structural_issues = _missing_dependency_issues(step_map)
        structural_issues.extend(_cycle_issues(step_map))
        grounding_issues, grounding_stats = _grounding_analysis(step_map, state)
        state_issues, state_stats = _state_transition_analysis(step_map, state)

        issues: list[str] = []
        issues.extend(structural_issues)
        issues.extend(grounding_issues)
        issues.extend(state_issues)

        step_count = len(plan.steps)
        completed_steps = _count_status(state, {"done", "skipped"}, step_map)
        failed_steps = _count_status(state, {"failed"}, step_map)
        score_components = {
            "task_success_rate": _task_success_rate(step_count, completed_steps),
            "plan_validity_rate": _plan_validity_rate(step_count, structural_issues),
            "state_transition_accuracy": _ratio(
                state_stats["matched_transitions"],
                state_stats["checked_transitions"],
                default=1.0,
            ),
            "grounding_coverage": _ratio(
                grounding_stats["grounded_executable_steps"],
                grounding_stats["executable_steps"],
                default=1.0,
            ),
            "replan_recovery_rate": _replan_recovery_rate(state, step_count, failed_steps),
            "search_efficiency": _search_efficiency(state.world_state.values),
        }
        score = sum(
            QUALITY_SCORE_WEIGHTS[name] * value
            for name, value in score_components.items()
        )

        details = {
            "quality_score_version": "composite-v1",
            "active_version": state.active_version,
            "step_count": step_count,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "grounded_steps": len(state.grounded_actions),
            "executable_steps": grounding_stats["executable_steps"],
            "grounded_executable_steps": grounding_stats["grounded_executable_steps"],
            "checked_state_transitions": state_stats["checked_transitions"],
            "matched_state_transitions": state_stats["matched_transitions"],
            "score_weights": dict(QUALITY_SCORE_WEIGHTS),
            "score_components": score_components,
        }
        return EvalResult(
            passed=not issues,
            score=round(score, 6),
            issues=tuple(issues),
            details=details,
        )


def _missing_dependency_issues(step_map: dict[str, PlanStep]) -> list[str]:
    issues: list[str] = []
    for step in step_map.values():
        for dep in step.deps:
            if dep not in step_map:
                issues.append(f"{step.step_id} depends on missing step {dep}")
    return issues


def _cycle_issues(step_map: dict[str, PlanStep]) -> list[str]:
    indegree = {step_id: 0 for step_id in step_map}
    children: dict[str, list[str]] = {step_id: [] for step_id in step_map}
    for step in step_map.values():
        for dep in step.deps:
            if dep in step_map:
                indegree[step.step_id] += 1
                children[dep].append(step.step_id)

    queue = deque(step_id for step_id, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        current = queue.popleft()
        visited += 1
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if visited != len(step_map):
        return ["plan contains a dependency cycle"]
    return []


def _grounding_analysis(
    step_map: dict[str, PlanStep],
    state: RunState,
) -> tuple[list[str], dict[str, int]]:
    issues: list[str] = []
    executable_steps = 0
    grounded_executable_steps = 0
    for step in step_map.values():
        if step.kind != "reason":
            executable_steps += 1
            if step.step_id in state.grounded_actions:
                grounded_executable_steps += 1
            else:
                issues.append(f"executable step {step.step_id} is not grounded to a capability")
    return issues, {
        "executable_steps": executable_steps,
        "grounded_executable_steps": grounded_executable_steps,
    }


def _state_transition_analysis(
    step_map: dict[str, PlanStep],
    state: RunState,
) -> tuple[list[str], dict[str, int]]:
    issues: list[str] = []
    checked_transitions = 0
    matched_transitions = 0
    for step_id, observed in state.observed_state_deltas.items():
        step = step_map.get(step_id)
        if step is None:
            continue
        for key, expected_value in step.expected_state_delta.set_values.items():
            if key not in observed.set_values:
                continue
            checked_transitions += 1
            if observed.set_values[key] == expected_value:
                matched_transitions += 1
            else:
                issues.append(
                    f"{step_id} observed state mismatch for {key}: expected {expected_value!r}, got {observed.set_values[key]!r}"
                )
    return issues, {
        "checked_transitions": checked_transitions,
        "matched_transitions": matched_transitions,
    }


def _count_status(state: RunState, statuses: set[str], step_map: dict[str, PlanStep]) -> int:
    return sum(1 for step_id in step_map if state.step_statuses.get(step_id) in statuses)


def _task_success_rate(step_count: int, completed_steps: int) -> float:
    return _ratio(completed_steps, step_count, default=0.0)


def _plan_validity_rate(step_count: int, structural_issues: list[str]) -> float:
    if step_count <= 0:
        return 0.0
    return _clamp(1.0 - (len(structural_issues) / step_count))


def _replan_recovery_rate(state: RunState, step_count: int, failed_steps: int) -> float:
    if state.active_version <= 1:
        return 1.0 if failed_steps == 0 else 0.0
    return _clamp(1.0 - (failed_steps / max(step_count, 1)))


def _search_efficiency(world_values: Mapping[str, object]) -> float:
    direct = world_values.get("_search_efficiency")
    if isinstance(direct, (int, float)):
        return _clamp(float(direct))

    metadata = world_values.get("_search")
    if not isinstance(metadata, Mapping):
        return 1.0

    efficiency = metadata.get("efficiency")
    if isinstance(efficiency, (int, float)):
        return _clamp(float(efficiency))

    expansions = metadata.get("expansions")
    budget = metadata.get("budget")
    if isinstance(expansions, (int, float)) and isinstance(budget, (int, float)) and budget > 0:
        return _clamp(1.0 - (float(expansions) / float(budget)))

    return 1.0


def _ratio(numerator: int, denominator: int, *, default: float) -> float:
    if denominator <= 0:
        return default
    return _clamp(numerator / denominator)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
