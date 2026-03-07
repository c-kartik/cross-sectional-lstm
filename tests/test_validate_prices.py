from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import validate_prices as vp


def test_validate_prices_flags_extreme_move_and_missing_cols(
    tmp_path: Path, monkeypatch
) -> None:
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)

    idx = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])

    # Missing "volume" on purpose.
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 150.0],
            "high": [101.0, 102.0, 151.0],
            "low": [99.0, 100.0, 149.0],
            "close": [100.0, 100.0, 150.0],  # 50% move on last day
        },
        index=idx,
    )
    df.to_parquet(prices_dir / "TEST.parquet")

    run_dir = tmp_path / "report_out"
    run_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(vp, "_report_dir", lambda: run_dir)
    monkeypatch.setattr(vp, "_expected_bdays", lambda s, e: pd.date_range(s, e, freq="B"))

    out_dir = vp.validate_prices(
        prices_dir=prices_dir,
        cfg=vp.QualityConfig(max_abs_return=0.3, missing_pct=0.5),
    )
    assert out_dir == run_dir

    issues = pd.read_csv(run_dir / "issues.csv")
    assert len(issues) == 1
    assert issues.loc[0, "flag_extreme"]
    assert issues.loc[0, "flag_missing_cols"]
