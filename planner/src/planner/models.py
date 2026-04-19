from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

StepKind = Literal["reason", "retrieve", "tool", "validate", "synthesize"]
StepStatus = Literal["pending", "ready", "running", "done", "failed", "skipped"]
RunStatus = Literal["active", "waiting_human", "completed", "failed", "cancelled"]


def _to_tuple(values: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(values)


def _to_dict(values: Mapping[str, Any] | None) -> dict[str, Any]:
    if not values:
        return {}
    return dict(values)


@dataclass(frozen=True)
class GoalSpec:
    user_query: str
    objective: str
    constraints: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()
    output_format: str = "text"
    ambiguity_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_query": self.user_query,
            "objective": self.objective,
            "constraints": list(self.constraints),
            "success_criteria": list(self.success_criteria),
            "output_format": self.output_format,
            "ambiguity_notes": list(self.ambiguity_notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GoalSpec:
        return cls(
            user_query=str(payload["user_query"]),
            objective=str(payload["objective"]),
            constraints=_to_tuple(payload.get("constraints")),
            success_criteria=_to_tuple(payload.get("success_criteria")),
            output_format=str(payload.get("output_format", "text")),
            ambiguity_notes=_to_tuple(payload.get("ambiguity_notes")),
        )


@dataclass(frozen=True)
class StateDelta:
    set_values: Mapping[str, Any] = field(default_factory=dict)
    remove_keys: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "set_values": dict(self.set_values),
            "remove_keys": list(self.remove_keys),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> StateDelta:
        if not payload:
            return cls()
        return cls(
            set_values=_to_dict(payload.get("set_values")),
            remove_keys=_to_tuple(payload.get("remove_keys")),
            notes=_to_tuple(payload.get("notes")),
        )

    def apply_to(self, snapshot: WorldStateSnapshot, *, updated_at: str) -> WorldStateSnapshot:
        values = dict(snapshot.values)
        values.update(self.set_values)
        for key in self.remove_keys:
            values.pop(key, None)
        return WorldStateSnapshot(values=values, revision=snapshot.revision + 1, updated_at=updated_at)


@dataclass(frozen=True)
class WorldStateSnapshot:
    values: Mapping[str, Any] = field(default_factory=dict)
    revision: int = 0
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": dict(self.values),
            "revision": self.revision,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> WorldStateSnapshot:
        if not payload:
            return cls()
        return cls(
            values=_to_dict(payload.get("values")),
            revision=int(payload.get("revision", 0)),
            updated_at=str(payload.get("updated_at", "")),
        )


@dataclass(frozen=True)
class PlanStep:
    step_id: str
    title: str
    kind: StepKind
    instructions: str
    deps: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()
    output_schema: Mapping[str, Any] = field(default_factory=dict)
    capability_hint: str | None = None
    expected_state_delta: StateDelta = field(default_factory=StateDelta)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "title": self.title,
            "kind": self.kind,
            "instructions": self.instructions,
            "deps": list(self.deps),
            "success_criteria": list(self.success_criteria),
            "output_schema": dict(self.output_schema),
            "capability_hint": self.capability_hint,
            "expected_state_delta": self.expected_state_delta.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PlanStep:
        return cls(
            step_id=str(payload["step_id"]),
            title=str(payload["title"]),
            kind=payload["kind"],
            instructions=str(payload["instructions"]),
            deps=_to_tuple(payload.get("deps")),
            success_criteria=_to_tuple(payload.get("success_criteria")),
            output_schema=_to_dict(payload.get("output_schema")),
            capability_hint=payload.get("capability_hint"),
            expected_state_delta=StateDelta.from_dict(payload.get("expected_state_delta")),
        )


@dataclass(frozen=True)
class PlanVersion:
    plan_id: str
    version: int
    parent_version: int | None
    reason: str
    steps: tuple[PlanStep, ...]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "version": self.version,
            "parent_version": self.parent_version,
            "reason": self.reason,
            "steps": [step.to_dict() for step in self.steps],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PlanVersion:
        return cls(
            plan_id=str(payload["plan_id"]),
            version=int(payload["version"]),
            parent_version=payload.get("parent_version"),
            reason=str(payload["reason"]),
            steps=tuple(PlanStep.from_dict(step) for step in payload.get("steps", [])),
            created_at=str(payload["created_at"]),
        )

    def step_map(self) -> dict[str, PlanStep]:
        return {step.step_id: step for step in self.steps}


@dataclass(frozen=True)
class FeedbackEvent:
    event_id: str
    run_id: str
    step_id: str | None
    source: str
    summary: str
    observed_state_delta: StateDelta = field(default_factory=StateDelta)
    new_open_questions: tuple[str, ...] = ()
    success: bool | None = None
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "step_id": self.step_id,
            "source": self.source,
            "summary": self.summary,
            "observed_state_delta": self.observed_state_delta.to_dict(),
            "new_open_questions": list(self.new_open_questions),
            "success": self.success,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FeedbackEvent:
        return cls(
            event_id=str(payload["event_id"]),
            run_id=str(payload["run_id"]),
            step_id=payload.get("step_id"),
            source=str(payload["source"]),
            summary=str(payload["summary"]),
            observed_state_delta=StateDelta.from_dict(payload.get("observed_state_delta")),
            new_open_questions=_to_tuple(payload.get("new_open_questions")),
            success=payload.get("success"),
            created_at=str(payload.get("created_at", "")),
        )


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    name: str
    kind: str
    uri: str
    produced_by_step: str
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "name": self.name,
            "kind": self.kind,
            "uri": self.uri,
            "produced_by_step": self.produced_by_step,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArtifactRef:
        return cls(
            artifact_id=str(payload["artifact_id"]),
            name=str(payload["name"]),
            kind=str(payload["kind"]),
            uri=str(payload["uri"]),
            produced_by_step=str(payload["produced_by_step"]),
            summary=str(payload.get("summary", "")),
        )


@dataclass
class RunState:
    run_id: str
    plan_id: str
    active_version: int
    status: RunStatus = "active"
    plan_versions: tuple[int, ...] = (1,)
    step_statuses: dict[str, StepStatus] = field(default_factory=dict)
    step_summaries: dict[str, str] = field(default_factory=dict)
    observed_state_deltas: dict[str, StateDelta] = field(default_factory=dict)
    grounded_actions: dict[str, str] = field(default_factory=dict)
    artifact_ids: list[str] = field(default_factory=list)
    feedback_ids: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    token_budget_remaining: int = 0
    tool_budget_remaining: int = 0
    final_response: str | None = None
    world_state: WorldStateSnapshot = field(default_factory=WorldStateSnapshot)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "plan_id": self.plan_id,
            "active_version": self.active_version,
            "status": self.status,
            "plan_versions": list(self.plan_versions),
            "step_statuses": dict(self.step_statuses),
            "step_summaries": dict(self.step_summaries),
            "observed_state_deltas": {
                key: delta.to_dict() for key, delta in self.observed_state_deltas.items()
            },
            "grounded_actions": dict(self.grounded_actions),
            "artifact_ids": list(self.artifact_ids),
            "feedback_ids": list(self.feedback_ids),
            "facts": list(self.facts),
            "assumptions": list(self.assumptions),
            "open_questions": list(self.open_questions),
            "token_budget_remaining": self.token_budget_remaining,
            "tool_budget_remaining": self.tool_budget_remaining,
            "final_response": self.final_response,
            "world_state": self.world_state.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunState:
        return cls(
            run_id=str(payload["run_id"]),
            plan_id=str(payload["plan_id"]),
            active_version=int(payload["active_version"]),
            status=payload.get("status", "active"),
            plan_versions=tuple(int(version) for version in payload.get("plan_versions", [1])),
            step_statuses=dict(payload.get("step_statuses", {})),
            step_summaries=dict(payload.get("step_summaries", {})),
            observed_state_deltas={
                key: StateDelta.from_dict(delta)
                for key, delta in payload.get("observed_state_deltas", {}).items()
            },
            grounded_actions=dict(payload.get("grounded_actions", {})),
            artifact_ids=list(payload.get("artifact_ids", [])),
            feedback_ids=list(payload.get("feedback_ids", [])),
            facts=list(payload.get("facts", [])),
            assumptions=list(payload.get("assumptions", [])),
            open_questions=list(payload.get("open_questions", [])),
            token_budget_remaining=int(payload.get("token_budget_remaining", 0)),
            tool_budget_remaining=int(payload.get("tool_budget_remaining", 0)),
            final_response=payload.get("final_response"),
            world_state=WorldStateSnapshot.from_dict(payload.get("world_state")),
        )


@dataclass(frozen=True)
class ContextSnapshot:
    goal: GoalSpec
    plan_version: PlanVersion
    current_step: PlanStep
    ready_steps: tuple[PlanStep, ...]
    world_state: WorldStateSnapshot
    recent_artifacts: tuple[ArtifactRef, ...]
    open_questions: tuple[str, ...]
    completed_summaries: Mapping[str, str]
    grounded_actions: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal.to_dict(),
            "plan_version": self.plan_version.to_dict(),
            "current_step": self.current_step.to_dict(),
            "ready_steps": [step.to_dict() for step in self.ready_steps],
            "world_state": self.world_state.to_dict(),
            "recent_artifacts": [artifact.to_dict() for artifact in self.recent_artifacts],
            "open_questions": list(self.open_questions),
            "completed_summaries": dict(self.completed_summaries),
            "grounded_actions": dict(self.grounded_actions),
        }


@dataclass(frozen=True)
class EvalResult:
    passed: bool
    score: float
    issues: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "issues": list(self.issues),
            "details": dict(self.details),
        }
