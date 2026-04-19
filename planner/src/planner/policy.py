from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .models import GoalSpec, PlanStep, StateDelta


class PlanningPolicy(Protocol):
    def interpret_goal(
        self,
        user_query: str,
        *,
        output_format: str = "text",
        success_criteria: tuple[str, ...] = (),
    ) -> GoalSpec:
        ...

    def build_initial_plan(self, goal: GoalSpec) -> tuple[PlanStep, ...]:
        ...


@dataclass
class RuleBasedPlanningPolicy:
    max_parts: int = 8

    def interpret_goal(
        self,
        user_query: str,
        *,
        output_format: str = "text",
        success_criteria: tuple[str, ...] = (),
    ) -> GoalSpec:
        cleaned = " ".join(user_query.split())
        objective = cleaned.rstrip(".")
        if not success_criteria:
            success_criteria = (
                "produce a complete response",
                "preserve dependencies between requested steps",
            )
        return GoalSpec(
            user_query=cleaned,
            objective=objective,
            constraints=tuple(_extract_constraints(cleaned)),
            success_criteria=success_criteria,
            output_format=output_format,
            ambiguity_notes=tuple(_extract_ambiguities(cleaned)),
        )

    def build_initial_plan(self, goal: GoalSpec) -> tuple[PlanStep, ...]:
        parts = _split_query(goal.user_query, max_parts=self.max_parts)
        if len(parts) <= 1:
            return (
                PlanStep(
                    step_id="analyze_request",
                    title="Analyze request",
                    kind="reason",
                    instructions=f"Interpret the request and identify required subtasks for: {goal.objective}",
                    success_criteria=("subtasks identified", "constraints captured"),
                    expected_state_delta=StateDelta(
                        set_values={"analysis_ready": True},
                        notes=("analysis complete",),
                    ),
                ),
                PlanStep(
                    step_id="execute_request",
                    title="Execute request",
                    kind="tool",
                    instructions=f"Carry out the main requested work for: {goal.objective}",
                    deps=("analyze_request",),
                    success_criteria=goal.success_criteria,
                    capability_hint=_capability_hint(goal.objective),
                    expected_state_delta=StateDelta(
                        set_values={"execution_ready": True},
                        notes=("execution complete",),
                    ),
                ),
                PlanStep(
                    step_id="validate_response",
                    title="Validate and summarize",
                    kind="validate",
                    instructions="Check that the result satisfies the request and produce the final summary.",
                    deps=("execute_request",),
                    success_criteria=("response checked", "final summary prepared"),
                    capability_hint="validator",
                    expected_state_delta=StateDelta(
                        set_values={"validated": True},
                        notes=("validation complete",),
                    ),
                ),
            )

        steps: list[PlanStep] = []
        previous_step_id: str | None = None
        for index, part in enumerate(parts, start=1):
            step_id = f"step_{index}"
            deps = (previous_step_id,) if previous_step_id else ()
            steps.append(
                PlanStep(
                    step_id=step_id,
                    title=_title_for_part(part, index),
                    kind=_kind_for_part(part),
                    instructions=part,
                    deps=deps,
                    success_criteria=(f"complete: {part}",),
                    capability_hint=_capability_hint(part),
                    expected_state_delta=StateDelta(
                        set_values={f"{step_id}_completed": True},
                        notes=(f"completed {part}",),
                    ),
                )
            )
            previous_step_id = step_id

        steps.append(
            PlanStep(
                step_id="validate_response",
                title="Validate and summarize",
                kind="validate",
                instructions="Review completed subtasks, verify dependency integrity, and prepare the final response.",
                deps=(previous_step_id,) if previous_step_id else (),
                success_criteria=("all subtasks checked", "final response ready"),
                capability_hint="validator",
                expected_state_delta=StateDelta(
                    set_values={"validated": True},
                    notes=("validation complete",),
                ),
            )
        )
        return tuple(steps)


def _extract_constraints(text: str) -> list[str]:
    matches: list[str] = []
    patterns = (
        r"\bmust\b[^.?!;]*",
        r"\bshould\b[^.?!;]*",
        r"\bwithout\b[^.?!;]*",
        r"\bwithin\b[^.?!;]*",
        r"\bby\b [^.?!;]*",
    )
    for pattern in patterns:
        matches.extend(match.group(0).strip() for match in re.finditer(pattern, text, flags=re.IGNORECASE))
    unique: list[str] = []
    seen: set[str] = set()
    for match in matches:
        lowered = match.lower()
        if lowered not in seen:
            seen.add(lowered)
            unique.append(match)
    return unique


def _extract_ambiguities(text: str) -> list[str]:
    notes: list[str] = []
    if re.search(r"\b(it|this|that|them)\b", text, flags=re.IGNORECASE):
        notes.append("query contains unresolved pronouns")
    if " or " in text.lower():
        notes.append("query may contain branching choices")
    return notes


def _split_query(text: str, *, max_parts: int) -> list[str]:
    raw_parts = re.split(
        r"\bthen\b|\band then\b|\bafter that\b|\bfinally\b|;|\n|[.?!](?=\s+[A-Z]|\s*$)",
        text,
    )
    parts = [part.strip(" ,.") for part in raw_parts if part.strip(" ,.")]
    return parts[:max_parts]


def _kind_for_part(text: str):
    lowered = text.lower()
    if any(token in lowered for token in ("find", "search", "look up", "retrieve", "inspect", "research")):
        return "retrieve"
    if any(token in lowered for token in ("check", "validate", "verify", "confirm", "test", "compare")):
        return "validate"
    if any(token in lowered for token in ("write", "draft", "summarize", "respond", "recommend")):
        return "synthesize"
    if any(token in lowered for token in ("run", "call", "update", "create", "build", "execute", "use")):
        return "tool"
    return "reason"


def _title_for_part(text: str, index: int) -> str:
    words = text.split()
    label = " ".join(words[:5]).strip()
    if not label:
        return f"Step {index}"
    return label[:1].upper() + label[1:]


def _capability_hint(text: str) -> str | None:
    lowered = text.lower()
    if any(token in lowered for token in ("search", "research", "find", "retrieve")):
        return "web_search"
    if any(token in lowered for token in ("write", "draft", "summarize", "respond", "recommend")):
        return "text_generation"
    if any(token in lowered for token in ("run", "execute", "build", "create", "update", "use")):
        return "workspace_action"
    if any(token in lowered for token in ("validate", "check", "verify", "test", "compare")):
        return "validator"
    return None
