from __future__ import annotations

from tool_index.rebuild import sweep_configs, ConfigPoint
from tool_index.rebuild.autotune import score_eval, DEFAULT_GRID
from tool_index.rebuild.eval_adapter import EvalQuery


def test_score_eval_blends_pos_and_neg() -> None:
    eval_set = [
        EvalQuery("q1", "t1", "positive", 0.9),
        EvalQuery("q2", "t2", "positive", 0.9),
        EvalQuery("q3", "t3", "negative", 0.9),
    ]
    retrieved = {"q1": ["t1"], "q2": ["tx"], "q3": ["t3", "ty"]}
    pos, neg, n = score_eval(eval_set, retrieved.get)
    assert pos == 0.5
    assert neg == 1.0
    assert n == 3


def test_sweep_picks_highest_objective() -> None:
    grid = [
        ConfigPoint(0.05, 0.10, 6),
        ConfigPoint(0.10, 0.15, 10),
    ]

    def score(point: ConfigPoint):
        if point.domain_threshold == 0.10:
            return 0.9, 0.1, 10
        return 0.5, 0.5, 10

    out = sweep_configs(grid, score, negative_penalty=0.5)
    assert out[0].point.domain_threshold == 0.10
    assert out[0].objective > out[1].objective


def test_default_grid_is_27_points() -> None:
    assert len(DEFAULT_GRID) == 27
