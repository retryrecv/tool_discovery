from __future__ import annotations

from planner import DEFAULT_EVAL_CASES_PATH, evaluate_cases, load_eval_cases


def test_eval_cases_cover_easy_to_complex():
    cases = load_eval_cases(DEFAULT_EVAL_CASES_PATH)

    assert len(cases) == 8
    assert [case.difficulty for case in cases] == [
        "easy",
        "easy",
        "medium",
        "medium",
        "hard",
        "hard",
        "complex",
        "complex",
    ]


def test_eval_cases_pass_with_current_planner(runtime):
    cases = load_eval_cases(DEFAULT_EVAL_CASES_PATH)
    results = evaluate_cases(
        cases,
        policy=runtime.policy,
        action_grounder=runtime.action_grounder,
    )

    assert all(result.passed for result in results)
    assert results[0].score >= results[-1].details["quality_floor"]
    assert all(result.details["step_count"] >= 3 for result in results)
