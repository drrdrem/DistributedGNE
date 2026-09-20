import numpy as np
import matplotlib.pyplot as plt

import src.MultiRobots.RobotsCovering.continuous.primal_dual_general as cpd


# ============================================================
# Multi-robot geometry
# ============================================================

T = {
    0: [0.0, 3.0, 0],
    1: [0.5, 3.5, 0],
    2: [1.0, 3.5, 0],

    3: [0.0, 0.0, 1],
    4: [0.5, 0.5, 1],
    5: [1.0, 0.0, 1],

    6: [3.0, 1.5, 2],
    7: [3.5, 2.0, 2],
    8: [4.0, 1.0, 2],
    9: [5.0, 3.0, 2],
}


G = {
    0: [0, 1, 2],
    1: [3, 4, 5],
    2: [6, 7, 8, 9],
}


colors = [
    "navy", "slateblue", "royalblue", "lightcoral", "peru", 
    "darkorange", "olive", "darkgreen", "seagreen", "teal",
]


# ============================================================
# Original shared half-space geometry
#
# For group G:
#   g^G(z) = -n_G^T center_G - c_G <= 0
#
# Therefore the shaded feasible side is
#   n_G^T p + c_G >= 0.
# ============================================================

line_coefficients = {
    0: {
        "coef": np.array([-0.6, 1.23]),
        "intercept": -2.2,
    },
    1: {
        "coef": np.array([-0.6, -1.2]),
        "intercept": 2.1,
    },
    2: {
        "coef": np.array([1.3, -0.1]),
        "intercept": -3.1,
    },
}


# ============================================================
# Cost weights
# ============================================================

rho = np.ones(len(T), dtype=float)


# ============================================================
# Individual robot constraints
#
# Each robot must remain inside the physical workspace.
# These are meaningful individual constraints h_v(z_v) <= 0.
#
# The shared half-space constraints above remain separate.
# ============================================================

local_bounds = {
    # group 0
    0: {"lo": [-0.35, 2.45], "hi": [0.95, 3.55]},
    1: {"lo": [-0.05, 2.85], "hi": [1.25, 3.95]},
    2: {"lo": [ 0.35, 2.85], "hi": [1.65, 3.95]},

    # group 1
    3: {"lo": [-0.35, -0.35], "hi": [0.95, 0.75]},
    4: {"lo": [-0.05,  0.05], "hi": [1.25, 1.15]},
    5: {"lo": [ 0.35, -0.35], "hi": [1.65, 0.75]},

    # group 2
    6: {"lo": [2.55, 0.95], "hi": [3.75, 2.05]},
    7: {"lo": [3.05, 1.45], "hi": [4.25, 2.55]},
    8: {"lo": [3.55, 0.45], "hi": [4.75, 1.55]},
    9: {"lo": [4.35, 2.45], "hi": [5.55, 3.55]},
}

# ============================================================
# 2-D spatial figure
# ============================================================

def plot_trajectory_2d(
    X,
    T,
    line_coefficients,
    G,
    colors,
    local_bounds=None,
    GNEs=None,
    lims=None,
    name="GeneralCostTrajectory2D",
):
    """
    2-D trajectory plot.

    Shows:
        - shared feasible half-planes,
        - shared-constraint boundaries,
        - individual/local box constraints,
        - target positions,
        - representative trajectories,
        - representative terminal positions,
        - terminal GNEs from multiple initializations.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    agents = list(T.keys())
    idx = {v: i for i, v in enumerate(agents)}

    Z, _, _ = cpd.split_state(X, T)

    if lims is None:
        lims = [
            [-0.5, 5.5],
            [-0.5, 4.5],
        ]

    xmin, xmax = lims[0]
    ymin, ymax = lims[1]

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.set_xlim(
        xmin, xmax,
    )

    ax.set_ylim(
        ymin, ymax,
    )

    ax.set_aspect(
        "equal", adjustable="box",
    )

    ax.grid(
        True, alpha=0.45,
    )

    ax.tick_params(
        labelbottom=False, labelleft=False,
    )

    # ========================================================
    # 1. Shared feasible half-planes
    # ========================================================

    xx, yy = np.meshgrid(
        np.linspace(
            xmin, xmax, 600,
        ),
        np.linspace(
            ymin, ymax, 600,
        ),
    )

    for group, members in G.items():

        color = colors[members[0]]
        coef = np.asarray(
            line_coefficients[group]["coef"],
            dtype=float,
        )

        intercept = float(
            line_coefficients[group]["intercept"]
        )

        # Feasible region:
        # n_G^T p + c_G >= 0
        #
        mask = (
            coef[0] * xx + coef[1] * yy + intercept
        ) >= 0.0

        rgba = np.zeros(
            xx.shape + (4,), dtype=float,
        )

        rgba[..., :3] = (
            plt.matplotlib.colors.to_rgb(
                color
            )
        )

        rgba[..., 3] = (
            0.10 * mask.astype(float)
        )

        ax.imshow(
            rgba,
            extent=[
                xmin, xmax, ymin, ymax,
            ],
            origin="lower",
            interpolation="nearest",
            zorder=0,
        )

    # ========================================================
    # 2. Shared-constraint boundaries
    # ========================================================

    x_vals = np.linspace(
        xmin, xmax, 500,
    )

    for group, members in G.items():

        color = colors[members[0]]
        coef = np.asarray(
            line_coefficients[group]["coef"],
            dtype=float,
        )

        intercept = float(
            line_coefficients[group]["intercept"]
        )

        a, b = coef

        if abs(b) > 1e-12:

            y_vals = -(
                a * x_vals + intercept
            ) / b

            ax.plot(
                x_vals, y_vals,
                "--",
                linewidth=2.0, alpha=0.55,
                color=color, zorder=1,
            )

        elif abs(a) > 1e-12:

            x_boundary = (
                -intercept / a
            )

            ax.axvline(
                x_boundary,
                linestyle="--",
                linewidth=2.0, alpha=0.55,
                color=color, zorder=1,
            )

    # ========================================================
    # 3. Individual/local box constraints
    # ========================================================

    if local_bounds is not None:

        # Avoid plotting exactly identical boxes many times.
        seen_boxes = set()

        for v in agents:

            lo = np.asarray(
                local_bounds[v]["lo"], dtype=float,
            )

            hi = np.asarray(
                local_bounds[v]["hi"], dtype=float,
            )

            box_key = (
                float(lo[0]), float(lo[1]),
                float(hi[0]), float(hi[1]),
            )

            if box_key in seen_boxes:
                continue

            seen_boxes.add(
                box_key
            )

            width = (
                hi[0] - lo[0]
            )

            height = (
                hi[1] - lo[1]
            )

            rect = Rectangle(
                (
                    lo[0], lo[1],
                ),
                width, height,
                fill=False,
                linestyle=":", linewidth=1.6,
                edgecolor=colors[idx[v]],
                alpha=0.55, zorder=2,
            )

            ax.add_patch(
                rect
            )

    # ========================================================
    # 4. Targets, trajectories, GNEs and final points
    # ========================================================

    target_size = 0.10

    for v in agents:

        i = idx[v]
        color = colors[i]
        tx = float(
            T[v][0]
        )
        ty = float(
            T[v][1]
        )

        # ----------------------------------------------------
        # Target square
        # ----------------------------------------------------

        square_x = [
            tx - target_size / 2.0,
            tx + target_size / 2.0,
            tx + target_size / 2.0,
            tx - target_size / 2.0,
            tx - target_size / 2.0,
        ]

        square_y = [
            ty - target_size / 2.0,
            ty - target_size / 2.0,
            ty + target_size / 2.0,
            ty + target_size / 2.0,
            ty - target_size / 2.0,
        ]

        ax.plot(
            square_x, square_y,
            color=color,
            linewidth=2.0,
            alpha=0.50, zorder=4,
        )

        # ----------------------------------------------------
        # Representative trajectory
        # ----------------------------------------------------

        ax.plot(
            Z[:, i, 0], Z[:, i, 1],
            "--",
            linewidth=1.8,
            alpha=0.22,
            color=color, zorder=3,
        )

        # ----------------------------------------------------
        # Terminal GNEs from different Lambda(0)
        # ----------------------------------------------------

        if GNEs is not None:

            pts = np.asarray(
                GNEs[v], dtype=float,
            )

            ax.scatter(
                pts[:, 0], pts[:, 1],
                marker="x",
                s=24,
                color=color,
                alpha=0.55, zorder=5,
            )

        # ----------------------------------------------------
        # Representative final position
        # ----------------------------------------------------

        ax.scatter(
            Z[-1, i, 0], Z[-1, i, 1],
            s=95,
            facecolors="none",
            edgecolors=color,
            linewidths=2.0, zorder=6,
        )

    plt.tight_layout()

    if name is not None:

        plt.savefig(
            f"{name}.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()

# ============================================================
# Numerical convergence figure
# ============================================================

def plot_convergence(
    t, X, T, G, rho,
    line_coefficients,
    local_bounds,
    *,
    name="GeneralCostConvergence",
):
    """
    One convergence figure with two numerical checks:

      left:
          ||dot y(t)|| evaluated directly from the implemented ODE;
      right:
          ||y(t)-y(T)||.

    The first plot verifies that the primal-dual vector field vanishes.
    The second verifies that the computed state approaches a point over
    the simulated horizon.
    """
    ode_norm = cpd.rhs_norm(
        t, X, T, G,
        rho,
        line_coefficients,
        local_bounds,
    )

    terminal_distance = np.linalg.norm(
        X - X[-1],
        axis=1,
    )

    floor = 1e-15

    fig, axes = plt.subplots(
        1, 2,
        figsize=(9.5, 3.5),
    )

    axes[0].semilogy(
        t,
        np.maximum(ode_norm, floor),
        linewidth=1.8,
    )
    axes[0].set_xlabel(r"$t$")
    axes[0].set_ylabel(r"$\|\dot y(t)\|$")
    axes[0].grid(True, alpha=0.35)

    axes[1].semilogy(
        t,
        np.maximum(terminal_distance, floor),
        linewidth=1.8,
    )
    axes[1].set_xlabel(r"$t$")
    axes[1].set_ylabel(r"$\|y(t)-y(T)\|$")
    axes[1].grid(True, alpha=0.35)

    plt.tight_layout()
    if name is not None:
        plt.savefig(
            f"{name}.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


# ============================================================
# Optional Delta plot
#
# Not needed for the two main paper figures, but kept here because
# Delta is important in the analysis and is useful for debugging.
# ============================================================

def plot_deltas(
    t, X, T, G,
    *, name=None,
):
    Delta, references = cpd.compute_group_deltas(X, T, G,)

    plt.figure(figsize=(7, 4))

    for (group, v), delta in Delta.items():
        if v == references[group]:
            continue

        plt.plot(
            t, delta,
            linewidth=1.5,
            label=rf"$\Delta_{{{v}}}$, group {group}",
        )

    plt.xlabel(r"$t$")
    plt.ylabel(r"$\Delta(t)$")
    plt.grid(True, alpha=0.30)
    plt.legend(fontsize=8)
    plt.tight_layout()

    if name is not None:
        plt.savefig(
            f"{name}.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


# ============================================================
# Multiple-initialization experiment
# ============================================================

def run_experiments_ctpd(
    T, G, rho,
    line_coefficients,
    local_bounds,
    *,
    seeds=(0, 1, 2, 3, 4, 5),
    position_seed=100,
    position_scale=0.25,
    lambda_scale=6.0,
    Tmax=120.0,
    TimeStamp=30_000,
    rtol=1e-7,
    atol=1e-9,
):
    """
    Run the same physical problem from the SAME initial robot positions
    but with different initial copied shared multipliers.

    This isolates the effect of multiplier initialization on the selected
    GNE.

    Returns
    -------
    t_first : time grid of the representative first run
    X_first : complete trajectory of the representative first run
    GNEs    : dict v -> terminal positions over all runs
    """
    agents = cpd._agent_order(T)
    n = len(agents)

    # --------------------------------------------------------
    # Fix z(0) across all runs.
    # --------------------------------------------------------
    rng_pos = np.random.default_rng(position_seed)

    targets = np.asarray(
        [T[v][:2] for v in agents],
        dtype=float,
    )

    Z0 = targets + rng_pos.normal(
        0.0,
        position_scale,
        size=(n, 2),
    )

    # Keep the common initial primal state inside the workspace.
    workspace_lo = np.asarray(
        local_bounds[agents[0]]["lo"],
        dtype=float,
    )
    workspace_hi = np.asarray(
        local_bounds[agents[0]]["hi"],
        dtype=float,
    )

    Z0 = np.clip(
        Z0,
        workspace_lo,
        workspace_hi,
    )

    GNEs = {
        v: []
        for v in agents
    }

    t_first = None
    X_first = None

    for run_index, seed in enumerate(seeds):
        y0 = cpd.make_init(
            T,
            seed=int(seed),
            fixed_positions=Z0,
            lambda_scale=lambda_scale,
            mu_scale=0.0,
        )

        t, X = cpd.solve(
            T, G, rho,
            line_coefficients,
            local_bounds,
            Tmax=Tmax,
            TimeStamp=TimeStamp,
            x_init=y0,
            rtol=rtol,
            atol=atol,
        )

        Z, _, _ = cpd.split_state(
            X, T,
        )

        for i, v in enumerate(agents):
            GNEs[v].append(
                Z[-1, i].copy()
            )

        if run_index == 0:
            t_first = t
            X_first = X

    return (
        t_first, X_first, GNEs,
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    t_first, X_first, GNEs = run_experiments_ctpd(
        T, G, rho,
        line_coefficients,
        local_bounds,
        seeds=(0, 1, 2, 3, 4, 5),
        Tmax=120.0,
        TimeStamp=30_000,
    )

    # Figure 1: spatial result
    plot_trajectory_2d(
        X_first, T,
        line_coefficients,
        G, colors,
        GNEs=GNEs,
        lims=[
            [-0.5, 5.5],
            [-0.5, 4.5],
        ],
        name="GeneralCostTrajectory2D",
    )

    # Figure 2: numerical convergence
    plot_convergence(
        t_first,
        X_first,
        T, G, rho,
        line_coefficients,
        local_bounds,
        name="GeneralCostConvergence",
    )
