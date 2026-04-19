from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .grounding import ActionGrounder
from .policy import PlanningPolicy

DEFAULT_EVAL_CASES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "eval_cases.jsonl"
)

BENCHMARK_SCORE_WEIGHTS = {
    "step_range": 0.20,
    "kind_coverage": 0.20,
    "term_coverage": 0.25,
    "constraint_recall": 0.15,
    "ambiguity_recall": 0.05,
    "grounding_coverage": 0.15,
}


@dataclass(frozen=True)
class PlannerCaseExpectations:
    min_steps: int
    max_steps: int
    required_kinds: tuple[str, ...] = ()
    required_terms: tuple[str, ...] = ()
    expected_constraints: tuple[str, ...] = ()
    expected_ambiguity_notes: tuple[str, ...] = ()
    min_grounded_steps: int = 0
    quality_floor: float = 0.70

    @classmethod
    def from_dict(cls, payload: dict) -> PlannerCaseExpectations:
        return cls(
            min_steps=int(payload["min_steps"]),
            max_steps=int(payload["max_steps"]),
            required_kinds=tuple(payload.get("required_kinds", [])),
            required_terms=tuple(payload.get("required_terms", [])),
            expected_constraints=tuple(payload.get("expected_constraints", [])),
            expected_ambiguity_notes=tuple(payload.get("expected_ambiguity_notes", [])),
            min_grounded_steps=int(payload.get("min_grounded_steps", 0)),
            quality_floor=float(payload.get("quality_floor", 0.70)),
        )


@dataclass(frozen=True)
class PlannerEvalCase:
    case_id: str
    difficulty: str
    query: str
    success_criteria: tuple[str, ...]
    expectations: PlannerCaseExpectations
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict) -> PlannerEvalCase:
        return cls(
            case_id=str(payload["id"]),
            difficulty=str(payload["difficulty"]),
            query=str(payload["query"]),
            success_criteria=tuple(payload.get("success_criteria", [])),
            expectations=PlannerCaseExpectations.from_dict(payload["expectations"]),
            notes=str(payload.get("notes", "")),
        )


@dataclass(frozen=True)
class PlannerCaseResult:
    case_id: str
    difficulty: str
    passed: bool
    score: float
    component_scores: dict[str, float]
    details: dict[str, object]


def load_eval_cases(path: str | Path = DEFAULT_EVAL_CASES_PATH) -> tuple[PlannerEvalCase, ...]:
    cases: list[PlannerEvalCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cases.append(PlannerEvalCase.from_dict(json.loads(line)))
    return tuple(cases)


def evaluate_case(
    case: PlannerEvalCase,
    *,
    policy: PlanningPolicy,
    action_grounder: ActionGrounder,
) -> PlannerCaseResult:
    goal = policy.interpret_goal(case.query, success_criteria=case.success_criteria)
    steps = policy.build_initial_plan(goal)

    step_text = " ".join(f"{step.title} {step.instructions}" for step in steps).lower()
    goal_constraints = [constraint.lower() for constraint in goal.constraints]
    ambiguity_notes = [note.lower() for note in goal.ambiguity_notes]
    kinds = {step.kind for step in steps}

    grounded_steps = 0
    executable_steps = 0
    for step in steps:
        if step.kind != "reason":
            executable_steps += 1
            if action_grounder.ground_step(step) is not None:
                grounded_steps += 1

    component_scores = {
        "step_range": 1.0
        if case.expectations.min_steps <= len(steps) <= case.expectations.max_steps
        else 0.0,
        "kind_coverage": _coverage(case.expectations.required_kinds, kinds),
        "term_coverage": _term_coverage(case.expectations.required_terms, step_text),
        "constraint_recall": _constraint_recall(case.expectations.expected_constraints, goal_constraints),
        "ambiguity_recall": _coverage(case.expectations.expected_ambiguity_notes, set(ambiguity_notes)),
        "grounding_coverage": _grounding_coverage(
            grounded_steps,
            executable_steps,
            case.expectations.min_grounded_steps,
        ),
    }
    score = sum(
        BENCHMARK_SCORE_WEIGHTS[name] * value
        for name, value in component_scores.items()
    )

    details = {
        "query": case.query,
        "goal_constraints": list(goal.constraints),
        "ambiguity_notes": list(goal.ambiguity_notes),
        "step_count": len(steps),
        "step_ids": [step.step_id for step in steps],
        "step_kinds": [step.kind for step in steps],
        "grounded_steps": grounded_steps,
        "executable_steps": executable_steps,
        "quality_floor": case.expectations.quality_floor,
    }
    return PlannerCaseResult(
        case_id=case.case_id,
        difficulty=case.difficulty,
        passed=score >= case.expectations.quality_floor,
        score=round(score, 6),
        component_scores=component_scores,
        details=details,
    )


def evaluate_cases(
    cases: tuple[PlannerEvalCase, ...],
    *,
    policy: PlanningPolicy,
    action_grounder: ActionGrounder,
) -> tuple[PlannerCaseResult, ...]:
    return tuple(
        evaluate_case(case, policy=policy, action_grounder=action_grounder)
        for case in cases
    )


def _coverage(expected: tuple[str, ...], actual: set[str]) -> float:
    if not expected:
        return 1.0
    hits = sum(1 for item in expected if item in actual)
    return _clamp(hits / len(expected))


def _term_coverage(terms: tuple[str, ...], step_text: str) -> float:
    if not terms:
        return 1.0
    hits = sum(1 for term in terms if term.lower() in step_text)
    return _clamp(hits / len(terms))


def _constraint_recall(expected_constraints: tuple[str, ...], goal_constraints: list[str]) -> float:
    if not expected_constraints:
        return 1.0
    hits = 0
    for expected in expected_constraints:
        lowered = expected.lower()
        if any(lowered in constraint for constraint in goal_constraints):
            hits += 1
    return _clamp(hits / len(expected_constraints))


def _grounding_coverage(
    grounded_steps: int,
    executable_steps: int,
    min_grounded_steps: int,
) -> float:
    if executable_steps == 0:
        return 1.0
    actual_coverage = grounded_steps / executable_steps
    if min_grounded_steps <= 0:
        return _clamp(actual_coverage)
    required_coverage = grounded_steps / min_grounded_steps
    return _clamp(min(actual_coverage, required_coverage))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
