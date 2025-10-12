import numpy as np

import src.MultiRobots.RobotsPlacement.continuous.primal_dual as cpd
import matplotlib.pyplot as plt

# -------------------- Experiment Harness (main) --------------------
def _block_size(use_box: bool) -> int:
    """Per-agent state block size."""
    return 6 if use_box else 2

def run_experiments(
    Agent, Center, C, rho, A,
    r=None,                              # dict or None (enables/disables μ boxes)
    seeds=(0, 1, 2, 3, 4),
    Tmax=120.0,
    TimeStamp=120_000,
    psi_mode="sum",                      # "sum" or "mean"
    plot_fn=None,                        # optional: def plot_fn(X, A, C, colors, r, GNEs, lims, name)
    plot_name="Trajectories",
    colors=None,
    lims=None,
    rtol=1e-6,
    atol=1e-8,
):
    """
    Runs multiple trajectories of the distributed primal–dual flow.

    Returns:
        t_first : (T,) time grid for the 1st run
        X_first : (T, dim) states for the 1st run
        GNEs    : dict v -> list of terminal z_v across runs (np.array shape (2,))
    """

    use_box = r is not None
    k = _block_size(use_box)
    agent_order = list(Agent.keys())
    n = len(agent_order)

    if colors is None:
        colors = [
            "royalblue", "slateblue", "mediumblue", "cadetblue", "darkslateblue",
            "deepskyblue", "cornflowerblue", "steelblue", "skyblue", "powderblue"
        ]
    if lims is None:
        lims = [[-4, 2], [-1.5, 2]]

    # ---------- First run (zero init) ----------
    L = k * n + sum(2 * len(Agent[v]) for v in Agent)   # add λ slots (2 per (v,c))
    y0_zero = np.zeros(L, dtype=float)

    t_first, X_first = cpd.solve(
        Agent, Center, C, rho, A, r,
        Tmax=Tmax, TimeStamp=TimeStamp,
        x_init=y0_zero, seed=seeds[0] if seeds else None,
        psi_mode=psi_mode,
        rtol=rtol, atol=atol
    )

    # Collect terminal positions per agent
    GNEs = {v: [X_first[-1, k*i:k*i+2]] for i, v in enumerate(agent_order)}

    # Quick state trace (optional, simple)
    plt.figure(figsize=(8, 3))
    plt.plot(t_first, X_first)
    plt.title("State traces (first run)")
    plt.xlabel("t")
    plt.ylabel("state components")
    plt.tight_layout()

    # ---------- Additional seeds ----------
    for s in seeds[1:]:
        y0 = cpd.make_init(Agent, r=r, seed=int(s))  # μ >= 0, λ free
        t, X = cpd.solve(
            Agent, Center, C, rho, A, r,
            Tmax=Tmax, TimeStamp=TimeStamp,
            x_init=y0, seed=int(s),
            psi_mode=psi_mode, rtol=rtol, atol=atol
        )

        # Per-run simple trace (optional)
        plt.figure(figsize=(8, 3))
        plt.plot(t, X)
        plt.title(f"State traces (seed={s})")
        plt.xlabel("t")
        plt.ylabel("state components")
        plt.tight_layout()

        # Append terminal positions
        for i, v in enumerate(agent_order):
            GNEs[v].append(X[-1, k*i:k*i+2])

    # ---------- Pretty plot using your helper (optional) ----------
    if plot_fn is not None:
        plot_fn(X_first, A, C, colors, r=r, GNEs=GNEs, lims=lims, name=plot_name)

    return t_first, X_first, GNEs