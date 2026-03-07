from __future__ import annotations

import numpy as np

from src.optimize.mean_variance import MVConfig, mean_variance_weights


def test_mean_variance_respects_basic_constraints() -> None:
    alpha = np.array([0.03, 0.02, 0.01], dtype=float)
    cov = np.eye(3) * 0.02
    cfg = MVConfig(max_weight=0.6, leverage=1.0, long_only=True)
    w = mean_variance_weights(alpha, cov, cfg=cfg)

    assert w.shape == (3,)
    assert np.isfinite(w).all()
    assert np.isclose(w.sum(), 1.0, atol=1e-4)
    assert (w >= -1e-8).all()
    assert (w <= 0.6 + 1e-6).all()


def test_rank_aware_min_weight_applies_to_top_mask() -> None:
    alpha = np.array([0.05, 0.04, 0.01, 0.0], dtype=float)
    cov = np.eye(4) * 0.01
    top_mask = np.array([True, True, False, False])
    cfg = MVConfig(
        max_weight=0.7,
        leverage=1.0,
        long_only=True,
        min_weight_top=0.1,
        risk_aversion=0.2,
    )
    w = mean_variance_weights(alpha, cov, top_mask=top_mask, cfg=cfg)

    assert np.isclose(w.sum(), 1.0, atol=1e-4)
    assert w[0] >= 0.1 - 1e-6
    assert w[1] >= 0.1 - 1e-6
