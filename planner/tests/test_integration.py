from __future__ import annotations

from planner import MockSymbolicSubplanner, PlanStep, StateDelta


def test_end_to_end_complex_session(runtime):
    runtime.symbolic_subplanner = MockSymbolicSubplanner(
        {"prepare deterministic checklist": ("collect", "rank", "return")}
    )

    session = runtime.start_run(
        "research two vendors, compare tradeoffs, then draft a recommendation",
        success_criteria=("recommendation produced", "tradeoffs compared"),
    )
    ready = runtime.next_ready_steps(session.run_id)
    assert ready[0].kind in {"retrieve", "reason"}

    runtime.mark_step_running(session.run_id, ready[0].step_id)
    runtime.complete_step(
        session.run_id,
        ready[0].step_id,
        summary="vendor research complete",
        artifacts=[("vendor_notes", {"vendors": ["A", "B"], "notes": ["A cheaper", "B faster"]})],
        observed_state_delta=StateDelta(set_values={f"{ready[0].step_id}_completed": True, "vendors_researched": True}),
    )
    runtime.record_feedback(
        session.run_id,
        step_id=ready[0].step_id,
        source="analysis",
        summary="support quality still unclear",
        observed_state_delta=StateDelta(set_values={"support_gap": True}),
        new_open_questions=("Which vendor offers stronger support?",),
        success=False,
    )

    deterministic_step = PlanStep(
        step_id="step_symbolic",
        title="Prepare deterministic checklist",
        kind="reason",
        instructions="prepare deterministic checklist",
        output_schema={"deterministic": True},
    )
    assert runtime.solve_deterministic_step(deterministic_step) == ("collect", "rank", "return")

    replanned = runtime.replan(
        session.run_id,
        (
            ready[0],
            PlanStep(
                step_id="step_support",
                title="Compare support quality",
                kind="validate",
                instructions="compare vendor support quality",
                deps=(ready[0].step_id,),
                capability_hint="validator",
                expected_state_delta=StateDelta(set_values={"support_compared": True}),
            ),
            PlanStep(
                step_id="step_write",
                title="Draft recommendation",
                kind="synthesize",
                instructions="draft the final recommendation",
                deps=("step_support",),
                capability_hint="text_generation",
                expected_state_delta=StateDelta(set_values={"draft_ready": True}),
            ),
            PlanStep(
                step_id="validate_response",
                title="Validate response",
                kind="validate",
                instructions="check the final answer",
                deps=("step_write",),
                capability_hint="validator",
                expected_state_delta=StateDelta(set_values={"validated": True}),
            ),
        ),
        reason="add explicit support comparison before final answer",
    )
    assert replanned.version == 2

    runtime.mark_step_running(session.run_id, "step_support")
    runtime.complete_step(
        session.run_id,
        "step_support",
        summary="support compared",
        observed_state_delta=StateDelta(set_values={"support_compared": True}),
    )
    runtime.mark_step_running(session.run_id, "step_write")
    runtime.complete_step(
        session.run_id,
        "step_write",
        summary="recommendation drafted",
        observed_state_delta=StateDelta(set_values={"draft_ready": True}),
    )
    runtime.mark_step_running(session.run_id, "validate_response")
    runtime.complete_step(
        session.run_id,
        "validate_response",
        summary="response validated",
        observed_state_delta=StateDelta(set_values={"validated": True}),
    )

    final_state, evaluation = runtime.finalize(
        session.run_id,
        final_response="Vendor B is the stronger recommendation because it balances speed and support.",
    )
    assert final_state.status == "completed"
    assert evaluation.passed is True
    assert evaluation.score == 1.0
    assert evaluation.details["score_components"]["task_success_rate"] == 1.0
    assert evaluation.details["score_components"]["grounding_coverage"] == 1.0
    assert final_state.final_response.startswith("Vendor B")
