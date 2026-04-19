from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .eval import PlanEvaluator
from .grounding import ActionGrounder, MockSymbolicSubplanner, RegistryActionGrounder, SymbolicSubplanner
from .models import ArtifactRef, ContextSnapshot, EvalResult, FeedbackEvent, GoalSpec, PlanStep, PlanVersion, RunState, StateDelta
from .policy import PlanningPolicy, RuleBasedPlanningPolicy
from .store import FilePlannerStore


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class PlanningSession:
    run_id: str
    goal: GoalSpec
    plan: PlanVersion
    state: RunState


class PlannerRuntime:
    def __init__(
        self,
        store: FilePlannerStore,
        *,
        policy: PlanningPolicy | None = None,
        action_grounder: ActionGrounder | None = None,
        symbolic_subplanner: SymbolicSubplanner | None = None,
        evaluator: PlanEvaluator | None = None,
    ) -> None:
        self.store = store
        self.policy = policy or RuleBasedPlanningPolicy()
        self.action_grounder = action_grounder or RegistryActionGrounder()
        self.symbolic_subplanner = symbolic_subplanner or MockSymbolicSubplanner()
        self.evaluator = evaluator or PlanEvaluator()

    def start_run(
        self,
        user_query: str,
        *,
        output_format: str = "text",
        success_criteria: tuple[str, ...] = (),
        token_budget_remaining: int = 0,
        tool_budget_remaining: int = 0,
    ) -> PlanningSession:
        goal = self.policy.interpret_goal(
            user_query,
            output_format=output_format,
            success_criteria=success_criteria,
        )
        steps = self.policy.build_initial_plan(goal)
        return self.create_run(
            goal,
            steps,
            reason="initial plan",
            token_budget_remaining=token_budget_remaining,
            tool_budget_remaining=tool_budget_remaining,
        )

    def create_run(
        self,
        goal: GoalSpec,
        steps: tuple[PlanStep, ...] | list[PlanStep],
        *,
        reason: str,
        token_budget_remaining: int = 0,
        tool_budget_remaining: int = 0,
    ) -> PlanningSession:
        run_id = f"run-{uuid4().hex[:12]}"
        plan_id = f"plan-{uuid4().hex[:12]}"
        created_at = _utc_now()
        plan = PlanVersion(
            plan_id=plan_id,
            version=1,
            parent_version=None,
            reason=reason,
            steps=tuple(steps),
            created_at=created_at,
        )
        state = RunState(
            run_id=run_id,
            plan_id=plan_id,
            active_version=1,
            plan_versions=(1,),
            step_statuses={step.step_id: "pending" for step in plan.steps},
            token_budget_remaining=token_budget_remaining,
            tool_budget_remaining=tool_budget_remaining,
        )
        self._ground_steps(plan, state)
        self.store.save_goal(run_id, goal)
        self.store.save_plan(run_id, plan)
        self.store.save_state(run_id, state)
        return PlanningSession(run_id=run_id, goal=goal, plan=plan, state=state)

    def get_session(self, run_id: str) -> PlanningSession:
        goal = self.store.load_goal(run_id)
        state = self.store.load_state(run_id)
        plan = self.store.load_plan(run_id, state.active_version)
        return PlanningSession(run_id=run_id, goal=goal, plan=plan, state=state)

    def get_active_plan(self, run_id: str) -> PlanVersion:
        state = self.store.load_state(run_id)
        return self.store.load_plan(run_id, state.active_version)

    def next_ready_steps(self, run_id: str) -> list[PlanStep]:
        session = self.get_session(run_id)
        step_map = session.plan.step_map()
        ready: list[PlanStep] = []
        for step in session.plan.steps:
            status = session.state.step_statuses.get(step.step_id, "pending")
            if status in {"done", "failed", "running", "skipped"}:
                continue
            if all(session.state.step_statuses.get(dep) == "done" for dep in step.deps):
                session.state.step_statuses[step.step_id] = "ready"
                ready.append(step)
        self.store.save_state(run_id, session.state)
        return ready

    def build_context(self, run_id: str, step_id: str, *, artifact_limit: int = 5) -> ContextSnapshot:
        session = self.get_session(run_id)
        current_step = session.plan.step_map()[step_id]
        ready_steps = tuple(self.next_ready_steps(run_id))
        recent_ids = session.state.artifact_ids[-artifact_limit:]
        recent_artifacts = tuple(self.store.load_artifacts(run_id, recent_ids))
        completed_summaries = {
            key: value
            for key, value in session.state.step_summaries.items()
            if session.state.step_statuses.get(key) == "done"
        }
        refreshed_state = self.store.load_state(run_id)
        return ContextSnapshot(
            goal=session.goal,
            plan_version=session.plan,
            current_step=current_step,
            ready_steps=ready_steps,
            world_state=refreshed_state.world_state,
            recent_artifacts=recent_artifacts,
            open_questions=tuple(refreshed_state.open_questions),
            completed_summaries=completed_summaries,
            grounded_actions=dict(refreshed_state.grounded_actions),
        )

    def record_feedback(
        self,
        run_id: str,
        *,
        step_id: str | None,
        source: str,
        summary: str,
        observed_state_delta: StateDelta | None = None,
        new_open_questions: tuple[str, ...] | list[str] = (),
        success: bool | None = None,
    ) -> FeedbackEvent:
        session = self.get_session(run_id)
        created_at = _utc_now()
        delta = observed_state_delta or StateDelta()
        event = FeedbackEvent(
            event_id=f"feedback-{uuid4().hex[:12]}",
            run_id=run_id,
            step_id=step_id,
            source=source,
            summary=summary,
            observed_state_delta=delta,
            new_open_questions=tuple(new_open_questions),
            success=success,
            created_at=created_at,
        )
        session.state.feedback_ids.append(event.event_id)
        session.state.open_questions.extend(question for question in event.new_open_questions if question not in session.state.open_questions)
        if delta.set_values or delta.remove_keys:
            session.state.world_state = delta.apply_to(session.state.world_state, updated_at=created_at)
        if step_id is not None:
            session.state.observed_state_deltas[step_id] = delta
        self.store.append_feedback(run_id, event)
        self.store.save_state(run_id, session.state)
        return event

    def mark_step_running(self, run_id: str, step_id: str) -> RunState:
        session = self.get_session(run_id)
        session.state.step_statuses[step_id] = "running"
        self.store.save_state(run_id, session.state)
        return session.state

    def complete_step(
        self,
        run_id: str,
        step_id: str,
        *,
        summary: str,
        artifacts: list[tuple[str, Any]] | None = None,
        facts: list[str] | None = None,
        assumptions: list[str] | None = None,
        open_questions: list[str] | None = None,
        observed_state_delta: StateDelta | None = None,
    ) -> RunState:
        session = self.get_session(run_id)
        session.state.step_statuses[step_id] = "done"
        session.state.step_summaries[step_id] = summary
        for item in facts or []:
            if item not in session.state.facts:
                session.state.facts.append(item)
        for item in assumptions or []:
            if item not in session.state.assumptions:
                session.state.assumptions.append(item)
        for item in open_questions or []:
            if item not in session.state.open_questions:
                session.state.open_questions.append(item)

        if observed_state_delta is not None:
            session.state.observed_state_deltas[step_id] = observed_state_delta
            if observed_state_delta.set_values or observed_state_delta.remove_keys:
                session.state.world_state = observed_state_delta.apply_to(session.state.world_state, updated_at=_utc_now())

        base_count = len(session.state.artifact_ids)
        for index, artifact_payload in enumerate(artifacts or [], start=1):
            name, payload = artifact_payload
            artifact_id = f"{step_id}-artifact-{base_count + index}"
            artifact_path = self.store.save_artifact(
                run_id,
                ArtifactRef(
                    artifact_id=artifact_id,
                    name=name,
                    kind="json",
                    uri=str(Path(self.store.root) / run_id / "artifacts" / f"{artifact_id}.json"),
                    produced_by_step=step_id,
                    summary=summary,
                ),
                payload,
            )
            del artifact_path
            session.state.artifact_ids.append(artifact_id)

        self.store.save_state(run_id, session.state)
        return session.state

    def fail_step(
        self,
        run_id: str,
        step_id: str,
        *,
        summary: str,
        open_questions: list[str] | None = None,
    ) -> RunState:
        session = self.get_session(run_id)
        session.state.step_statuses[step_id] = "failed"
        session.state.step_summaries[step_id] = summary
        for item in open_questions or []:
            if item not in session.state.open_questions:
                session.state.open_questions.append(item)
        self.store.save_state(run_id, session.state)
        return session.state

    def replan(
        self,
        run_id: str,
        steps: tuple[PlanStep, ...] | list[PlanStep],
        *,
        reason: str,
    ) -> PlanVersion:
        session = self.get_session(run_id)
        new_version = session.state.active_version + 1
        plan = PlanVersion(
            plan_id=session.plan.plan_id,
            version=new_version,
            parent_version=session.state.active_version,
            reason=reason,
            steps=tuple(steps),
            created_at=_utc_now(),
        )
        old_statuses = dict(session.state.step_statuses)
        old_summaries = dict(session.state.step_summaries)
        old_deltas = dict(session.state.observed_state_deltas)
        old_grounded = dict(session.state.grounded_actions)

        session.state.active_version = new_version
        session.state.plan_versions = (*session.state.plan_versions, new_version)
        session.state.step_statuses = {}
        session.state.step_summaries = {}
        session.state.observed_state_deltas = {}
        session.state.grounded_actions = {}

        for step in plan.steps:
            previous_status = old_statuses.get(step.step_id, "pending")
            if previous_status in {"done", "skipped"}:
                session.state.step_statuses[step.step_id] = previous_status
                if step.step_id in old_summaries:
                    session.state.step_summaries[step.step_id] = old_summaries[step.step_id]
                if step.step_id in old_deltas:
                    session.state.observed_state_deltas[step.step_id] = old_deltas[step.step_id]
                if step.step_id in old_grounded:
                    session.state.grounded_actions[step.step_id] = old_grounded[step.step_id]
            else:
                session.state.step_statuses[step.step_id] = "pending"

        self._ground_steps(plan, session.state)
        self.store.save_plan(run_id, plan)
        self.store.save_state(run_id, session.state)
        return plan

    def solve_deterministic_step(self, step: PlanStep) -> tuple[str, ...]:
        if not step.output_schema.get("deterministic"):
            return ()
        return self.symbolic_subplanner.solve(step.instructions)

    def finalize(self, run_id: str, *, final_response: str) -> tuple[RunState, EvalResult]:
        session = self.get_session(run_id)
        evaluation = self.evaluator.evaluate(session.plan, session.state)
        session.state.status = "completed" if evaluation.passed else "failed"
        session.state.final_response = final_response
        self.store.save_state(run_id, session.state)
        return session.state, evaluation

    def _ground_steps(self, plan: PlanVersion, state: RunState) -> None:
        for step in plan.steps:
            grounded = self.action_grounder.ground_step(step)
            if grounded is not None:
                state.grounded_actions[step.step_id] = grounded.capability_id
