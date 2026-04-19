from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from planner import GoalSpec, PlanStep, PlanVersion, RunState, StateDelta, WorldStateSnapshot


def test_models_round_trip_and_immutability():
    goal = GoalSpec(
        user_query="research vendors then recommend one",
        objective="research vendors then recommend one",
        constraints=("must cite sources",),
        success_criteria=("recommendation produced",),
        ambiguity_notes=("query may contain branching choices",),
    )
    step = PlanStep(
        step_id="step_1",
        title="Research vendors",
        kind="retrieve",
        instructions="research two vendors",
        success_criteria=("vendors researched",),
        capability_hint="web_search",
        expected_state_delta=StateDelta(set_values={"vendors_researched": True}),
    )
    plan = PlanVersion(
        plan_id="plan-1",
        version=1,
        parent_version=None,
        reason="initial",
        steps=(step,),
        created_at="2026-04-18T00:00:00+00:00",
    )
    state = RunState(
        run_id="run-1",
        plan_id="plan-1",
        active_version=1,
        world_state=WorldStateSnapshot(values={"vendors_researched": True}, revision=1, updated_at="now"),
        observed_state_deltas={"step_1": StateDelta(set_values={"vendors_researched": True})},
    )

    assert GoalSpec.from_dict(goal.to_dict()) == goal
    assert PlanStep.from_dict(step.to_dict()) == step
    assert PlanVersion.from_dict(plan.to_dict()) == plan
    assert RunState.from_dict(state.to_dict()) == state

    with pytest.raises(FrozenInstanceError):
        step.title = "changed"

    with pytest.raises(FrozenInstanceError):
        plan.version = 2
