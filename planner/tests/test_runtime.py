from __future__ import annotations

from planner import PlanStep, StateDelta


def test_runtime_step_lifecycle_context_feedback_and_replan(runtime):
    session = runtime.start_run(
        "research two vendors, compare tradeoffs, then draft a recommendation",
        token_budget_remaining=1000,
        tool_budget_remaining=5,
    )

    ready = runtime.next_ready_steps(session.run_id)
    assert ready
    first_step = ready[0]
    runtime.mark_step_running(session.run_id, first_step.step_id)

    completed_state = runtime.complete_step(
        session.run_id,
        first_step.step_id,
        summary="researched vendors",
        artifacts=[("vendors", {"vendors": ["A", "B"]})],
        facts=["vendor A is cheaper"],
        observed_state_delta=StateDelta(set_values={f"{first_step.step_id}_completed": True, "vendors_researched": True}),
    )
    assert completed_state.step_statuses[first_step.step_id] == "done"
    assert completed_state.artifact_ids

    context = runtime.build_context(session.run_id, "validate_response", artifact_limit=1)
    assert context.recent_artifacts
    assert first_step.step_id in context.completed_summaries

    feedback = runtime.record_feedback(
        session.run_id,
        step_id=first_step.step_id,
        source="tool",
        summary="need better differentiation on support",
        observed_state_delta=StateDelta(set_values={"needs_support_comparison": True}),
        new_open_questions=("Which vendor has better support?",),
        success=False,
    )
    assert feedback.event_id.startswith("feedback-")

    plan_v2 = runtime.replan(
        session.run_id,
        (
            PlanStep(
                step_id=first_step.step_id,
                title=first_step.title,
                kind=first_step.kind,
                instructions=first_step.instructions,
                success_criteria=first_step.success_criteria,
                capability_hint=first_step.capability_hint,
                expected_state_delta=first_step.expected_state_delta,
            ),
            PlanStep(
                step_id="step_support",
                title="Compare support",
                kind="validate",
                instructions="compare vendor support quality",
                deps=(first_step.step_id,),
                capability_hint="validator",
                expected_state_delta=StateDelta(set_values={"support_compared": True}),
            ),
            PlanStep(
                step_id="validate_response",
                title="Validate and summarize",
                kind="validate",
                instructions="produce final answer",
                deps=("step_support",),
                capability_hint="validator",
                expected_state_delta=StateDelta(set_values={"validated": True}),
            ),
        ),
        reason="need explicit support comparison",
    )
    assert plan_v2.version == 2

    session_v2 = runtime.get_session(session.run_id)
    assert session_v2.state.active_version == 2
    assert session_v2.state.step_statuses[first_step.step_id] == "done"
    assert "Which vendor has better support?" in session_v2.state.open_questions
