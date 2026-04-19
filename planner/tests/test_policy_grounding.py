from __future__ import annotations

from planner import CapabilitySpec, MockSymbolicSubplanner, PlanStep, RegistryActionGrounder, RuleBasedPlanningPolicy


def test_policy_decomposes_complex_query():
    policy = RuleBasedPlanningPolicy()
    goal = policy.interpret_goal(
        "research two vendors, compare tradeoffs, then draft a recommendation without changing the budget",
    )
    steps = policy.build_initial_plan(goal)

    assert goal.objective.startswith("research two vendors")
    assert any("without changing the budget" in constraint.lower() for constraint in goal.constraints)
    assert len(steps) >= 3
    assert steps[-1].step_id == "validate_response"
    assert steps[1].deps == (steps[0].step_id,)


def test_action_grounder_matches_capability_and_symbolic_solver_is_stable():
    grounder = RegistryActionGrounder(
        capabilities=(
            CapabilitySpec("web_search", "Search for vendor information", ("research", "search")),
            CapabilitySpec("validator", "Compare and validate results", ("compare", "validate")),
        )
    )
    step = PlanStep(
        step_id="step_1",
        title="Research vendors",
        kind="retrieve",
        instructions="search for vendor information",
        capability_hint="web_search",
    )
    grounded = grounder.ground_step(step)
    assert grounded is not None
    assert grounded.capability_id == "web_search"

    unmatched = PlanStep(
        step_id="step_2",
        title="Unknown",
        kind="tool",
        instructions="operate a crane in a warehouse",
    )
    assert grounder.ground_step(unmatched) is None

    symbolic = MockSymbolicSubplanner({"sort deterministic checklist": ("collect", "sort", "return")})
    assert symbolic.solve("sort deterministic checklist") == ("collect", "sort", "return")
    assert symbolic.solve("sort deterministic checklist") == ("collect", "sort", "return")
