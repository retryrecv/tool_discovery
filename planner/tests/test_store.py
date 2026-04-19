from __future__ import annotations

from planner import ArtifactRef, FeedbackEvent, FilePlannerStore, GoalSpec, PlanStep, PlanVersion, RunState, StateDelta


def test_file_store_round_trip(tmp_path):
    store = FilePlannerStore(tmp_path / "runs")
    goal = GoalSpec(user_query="research", objective="research")
    plan = PlanVersion(
        plan_id="plan-1",
        version=1,
        parent_version=None,
        reason="initial",
        steps=(PlanStep(step_id="step_1", title="Research", kind="retrieve", instructions="research"),),
        created_at="2026-04-18T00:00:00+00:00",
    )
    state = RunState(run_id="run-1", plan_id="plan-1", active_version=1)
    artifact = ArtifactRef(
        artifact_id="artifact-1",
        name="vendors",
        kind="json",
        uri=str(tmp_path / "artifact-1.json"),
        produced_by_step="step_1",
        summary="vendor list",
    )
    feedback = FeedbackEvent(
        event_id="feedback-1",
        run_id="run-1",
        step_id="step_1",
        source="tool",
        summary="research complete",
        observed_state_delta=StateDelta(set_values={"vendors_researched": True}),
        created_at="2026-04-18T00:00:00+00:00",
    )

    store.save_goal("run-1", goal)
    store.save_plan("run-1", plan)
    store.save_state("run-1", state)
    store.save_artifact("run-1", artifact, {"vendors": ["A", "B"]})
    store.append_feedback("run-1", feedback)

    assert store.load_goal("run-1") == goal
    assert store.load_plan("run-1", 1) == plan
    assert store.load_state("run-1") == state
    loaded_artifact, payload = store.load_artifact("run-1", "artifact-1")
    assert loaded_artifact == artifact
    assert payload == {"vendors": ["A", "B"]}
    assert store.load_feedback("run-1") == [feedback]
    assert store.list_plan_versions("run-1") == [1]
