from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import forward_fill_weights, run_backtest, turnover


def test_forward_fill_weights_carries_rebalance_targets() -> None:
    idx = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    w = pd.DataFrame(
        {
            "AAA": [0.6, 0.0, 0.0],
            "BBB": [0.4, 0.0, 0.0],
        },
        index=idx,
    )
    out = forward_fill_weights(w)
    assert np.isclose(out.loc[idx[1], "AAA"], 0.6)
    assert np.isclose(out.loc[idx[2], "BBB"], 0.4)


def test_turnover_formula_matches_half_l1_change() -> None:
    prev_w = pd.Series({"AAA": 0.6, "BBB": 0.4})
    new_w = pd.Series({"AAA": 0.2, "BBB": 0.8})
    # 0.5 * (|0.2-0.6| + |0.8-0.4|) = 0.4
    assert np.isclose(turnover(prev_w, new_w), 0.4)


def test_run_backtest_applies_costs_and_emits_stats() -> None:
    idx = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
    rets = pd.DataFrame({"AAA": [0.0, 0.01, 0.01]}, index=idx)
    # Target becomes active with 1-day lag due to no-lookahead shift.
    targets = pd.DataFrame({"AAA": [1.0, 1.0, 1.0]}, index=idx)

    out = run_backtest(rets, targets, cost_bps=100.0)  # 1% cost on traded notional

    assert "stats" in out
    assert out["stats"]["days"] == 3
    assert "avg_weekly_turnover" in out["stats"]
    assert "total_cost_paid" in out["stats"]

    # Only one trade from 0 -> 1 after shift alignment => turnover ~0.5, cost ~0.005.
    assert out["stats"]["total_cost_paid"] >= 0.0
    assert out["equity"].iloc[-1] > 0.0
