from __future__ import annotations

import pytest

from planner import BestFirstSearchPolicy, PlanEvaluator, PlanStep, PlanVersion, RunState, SearchNode, StateDelta


def test_best_first_search_picks_higher_reward_branch():
    policy = BestFirstSearchPolicy(max_expansions=4, beam_width=2)
    root = SearchNode(node_id="root", state={"value": 0}, score=0.1)

    graph = {
        "root": (
            SearchNode(node_id="a", state={"value": 1}, score=0.4, parent_id="root"),
            SearchNode(node_id="b", state={"value": 2}, score=0.8, parent_id="root"),
        ),
        "a": (SearchNode(node_id="a1", state={"value": 3}, score=0.5, parent_id="a"),),
        "b": (SearchNode(node_id="b1", state={"value": 4}, score=0.9, parent_id="b"),),
        "b1": (),
    }

    result = policy.search(root, lambda node: graph.get(node.node_id, ()), lambda node: node.node_id == "b1")

    assert result.best_node.node_id == "b1"
    assert [node.node_id for node in result.path] == ["root", "b", "b1"]
    assert result.expansions <= 4


def test_evaluator_rejects_missing_grounding_and_state_mismatch():
    plan = PlanVersion(
        plan_id="plan-1",
        version=1,
        parent_version=None,
        reason="initial",
        steps=(
            PlanStep(
                step_id="step_1",
                title="Execute work",
                kind="tool",
                instructions="do work",
                expected_state_delta=StateDelta(set_values={"done": True}),
            ),
        ),
        created_at="2026-04-18T00:00:00+00:00",
    )
    state = RunState(
        run_id="run-1",
        plan_id="plan-1",
        active_version=1,
        observed_state_deltas={"step_1": StateDelta(set_values={"done": False})},
    )

    result = PlanEvaluator().evaluate(plan, state)

    assert result.passed is False
    assert any("not grounded" in issue for issue in result.issues)
    assert any("observed state mismatch" in issue for issue in result.issues)
    assert result.score == pytest.approx(0.35)
    assert result.details["quality_score_version"] == "composite-v1"
    assert result.details["score_components"]["task_success_rate"] == 0.0
    assert result.details["score_components"]["grounding_coverage"] == 0.0
    assert result.details["score_components"]["state_transition_accuracy"] == 0.0
