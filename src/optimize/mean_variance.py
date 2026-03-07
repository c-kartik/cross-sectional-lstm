from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np


@dataclass(frozen=True)
class MVConfig:
    risk_aversion: float = 1.0
    turnover_penalty: float = 1.0
    max_weight: float = 0.1
    long_only: bool = True
    leverage: float = 1.0  # sum of weights
    solver: str | None = "OSQP"
    solver_max_iters: int = 20000
    solver_eps: float = 1e-6
    cov_jitter: float = 1e-6
    alpha_scale: float = 1.0
    cov_shrinkage: float = 0.0
    min_weight_top: float = 0.0


def mean_variance_weights(
    alpha: np.ndarray,
    cov: np.ndarray,
    w_prev: np.ndarray | None = None,
    top_mask: np.ndarray | None = None,
    cfg: MVConfig | None = None,
) -> np.ndarray:
    if cfg is None:
        cfg = MVConfig()

    n = int(len(alpha))
    if n == 0:
        return np.array([])

    alpha = np.asarray(alpha, dtype=float).reshape(-1) * float(cfg.alpha_scale)
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    if cfg.cov_shrinkage and cfg.cov_shrinkage > 0:
        lam = float(cfg.cov_shrinkage)
        diag = np.diag(np.diag(cov))
        cov = (1.0 - lam) * cov + lam * diag
    if cfg.cov_jitter and cfg.cov_jitter > 0:
        cov = cov + np.eye(cov.shape[0]) * cfg.cov_jitter

    if w_prev is None:
        w_prev = np.zeros(n, dtype=float)
    else:
        w_prev = np.asarray(w_prev, dtype=float).reshape(-1)

    w = cp.Variable(n)
    risk = cp.quad_form(w, cov)
    turnover = cp.norm1(w - w_prev)
    objective = cp.Maximize(alpha @ w - cfg.risk_aversion * risk - cfg.turnover_penalty * turnover)

    constraints = [cp.sum(w) == cfg.leverage]
    if cfg.long_only:
        constraints.append(w >= 0)
    if cfg.max_weight is not None:
        constraints.append(w <= cfg.max_weight)
    if cfg.min_weight_top and cfg.min_weight_top > 0 and top_mask is not None:
        top_mask = np.asarray(top_mask, dtype=bool).reshape(-1)
        if top_mask.any():
            constraints.append(w[top_mask] >= cfg.min_weight_top)

    prob = cp.Problem(objective, constraints)

    try:
        if cfg.solver:
            prob.solve(
                solver=cfg.solver,
                verbose=False,
                max_iter=cfg.solver_max_iters,
                eps_abs=cfg.solver_eps,
                eps_rel=cfg.solver_eps,
            )
        else:
            prob.solve(verbose=False)
    except Exception:
        return _fallback_weights(alpha, cfg)

    if w.value is None or not np.all(np.isfinite(w.value)):
        return _fallback_weights(alpha, cfg)

    return np.asarray(w.value, dtype=float).reshape(-1)


def _fallback_weights(alpha: np.ndarray, cfg: MVConfig) -> np.ndarray:
    a = np.asarray(alpha, dtype=float)
    if cfg.long_only:
        a = np.clip(a, 0.0, None)
    if a.sum() == 0:
        w = np.ones_like(a) / len(a)
    else:
        w = a / a.sum()
    w = np.minimum(w, cfg.max_weight) if cfg.max_weight is not None else w
    if w.sum() > 0:
        w = w / w.sum() * cfg.leverage
    return w
