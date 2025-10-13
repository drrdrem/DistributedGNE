import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Iterable

import src.MultiRobots.RobotaCovering.continuous.primal_dual as cpd

# ------------------------------------------------------------
# EXPERIMENT RUNNER (mirrors your multi-seed utility)
# ------------------------------------------------------------
def run_experiments_ctpd(
    T: Dict[int, Iterable[float]],
    G: Dict[int, List[int]],
    rho: Dict[int, float],
    line_coefficients: Dict[int, Dict[str, Iterable[float]]],
    *,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    Tmax: float = 120.0,
    TimeStamp: int = 120_000,
    plot_fn=None,                 # optional: def plot_fn(X_first, T, G, colors, GNEs, lims, name)
    plot_name: str = "Trajectories",
    colors: Optional[List[str]] = None,
    lims: Optional[List[List[float]]] = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    eps: float = 0.0,
    dynamics_fn=None,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[np.ndarray]]]:
    """
    Runs multiple trajectories of the continuous-time primal–dual flow.

    Returns:
        t_first : (T,) time grid for the 1st run
        X_first : (T, dim) states for the 1st run
        GNEs    : dict v -> list of terminal z_v across runs (np.array shape (2,))
    """
    agents = cpd._agent_order(T)
    n, _ = cpd._dim_counts(T)
    k = 2  # per-agent position block size

    if colors is None:
        colors = [
            "royalblue", "slateblue", "mediumblue", "cadetblue", "darkslateblue",
            "deepskyblue", "cornflowerblue", "steelblue", "skyblue", "powderblue"
        ]
    if lims is None:
        # [xlims, ylims] for potential plotting helpers
        all_T = np.asarray([T[v][:2] for v in agents], dtype=float)
        lo = (all_T.min(axis=0) - 1.5).tolist()
        hi = (all_T.max(axis=0) + 1.5).tolist()
        lims = [[lo[0], hi[0]], [lo[1], hi[1]]]

    # ---------- First run (zero init) ----------
    L = 3 * n  # 2n positions + n lambdas
    y0_zero = np.zeros(L, dtype=float)

    t_first, X_first = cpd.solve(
        T, G, rho, line_coefficients,
        Tmax=Tmax, TimeStamp=TimeStamp,
        x_init=y0_zero, seed=(seeds[0] if seeds else None),
        eps=eps, rtol=rtol, atol=atol, dynamics_fn=dynamics_fn
    )

    # Collect terminal positions per agent
    GNEs = {v: [X_first[-1, k*i:k*i+2].copy()] for i, v in enumerate(agents)}

    # Quick state trace (optional)
    plt.figure(figsize=(8, 3))
    plt.plot(t_first, X_first)
    plt.title("State traces (first run)")
    plt.xlabel("t")
    plt.ylabel("state components")
    plt.tight_layout()

    # ---------- Additional seeds ----------
    for s in list(seeds)[1:]:
        y0 = cpd.cpdmake_init(T, seed=int(s), init_mode="around_targets")
        cpd.ensure_lambda_nonneg_inplace(y0, T)

        t, X = cpd.solve(
            T, G, rho, line_coefficients,
            Tmax=Tmax, TimeStamp=TimeStamp,
            x_init=y0, seed=int(s),
            eps=eps, rtol=rtol, atol=atol, dynamics_fn=dynamics_fn
        )

        # Per-run simple trace (optional)
        plt.figure(figsize=(8, 3))
        plt.plot(t, X)
        plt.title(f"State traces (seed={s})")
        plt.xlabel("t")
        plt.ylabel("state components")
        plt.tight_layout()

        # Append terminal positions
        for i, v in enumerate(agents):
            GNEs[v].append(X[-1, k*i:k*i+2].copy())

    # ---------- Pretty plot using your helper (optional) ----------
    if plot_fn is not None:
        plot_fn(X_first, T, G, colors, GNEs=GNEs, lims=lims, name=plot_name)

    return t_first, X_first, GNEs

