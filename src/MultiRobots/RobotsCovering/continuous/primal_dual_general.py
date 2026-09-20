import numpy as np
from typing import Dict, Iterable, List, Optional, Tuple
from scipy.integrate import solve_ivp


# ============================================================
# Individual constraint
#   h_v(z_v) = C_v z_v - d_v <= 0
#
# Here each robot has a rectangular operating region
#   xmin <= x_v <= xmax,
#   ymin <= y_v <= ymax.
# ============================================================

LOCAL_C = np.array(
    [
        [1.0, 0.0], [-1.0, 0.0],
        [0.0, 1.0], [0.0, -1.0],
    ],
    dtype=float,
)


def local_C_d(bounds_v):
    """
    Convert
        bounds_v = {
            "lo": [xmin, ymin],
            "hi": [xmax, ymax],
        }

    into
        h_v(z_v) = C_v z_v - d_v <= 0.
    """
    lo = np.asarray(bounds_v["lo"], dtype=float)
    hi = np.asarray(bounds_v["hi"], dtype=float)

    d_v = np.array(
        [
            hi[0], -lo[0],
            hi[1], -lo[1],
        ],
        dtype=float,
    )

    return LOCAL_C, d_v


# ============================================================
# State layout
#
# y = [z, Lambda, mu]
#
# z      : 2N entries
# Lambda : N entries
# mu     : 4N entries
#
# total dimension = 7N
# ============================================================

def _agent_order(T: Dict[int, Iterable[float]]) -> List[int]:
    return list(T.keys())


def _dim_counts(T: Dict[int, Iterable[float]]) -> Tuple[int, int]:
    n = len(T)
    return n, 7 * n


def _lambda_start_index(T) -> int:
    return 2 * len(T)


def _mu_start_index(T) -> int:
    return 3 * len(T)


def ensure_dual_nonneg_inplace(y0: np.ndarray, T) -> None:
    """Project the initial Lambda and mu components to R_+."""
    dual_start = _lambda_start_index(T)
    y0[dual_start:] = np.maximum(y0[dual_start:], 0.0)


def _rho_value(rho, v, i) -> float:
    """
    Support either

        rho = {v: rho_v}

    or a list/array in agent order.
    """
    if isinstance(rho, dict):
        return float(rho[v])

    return float(np.asarray(rho, dtype=float).reshape(-1)[i])


def _projected_rate(residual: float, multiplier: float, eps: float = 0.0) -> float:
    """
    Implements
        [residual]^+_multiplier.

    For multiplier > 0, return residual.
    At the boundary multiplier = 0, only positive residuals are allowed.
    """
    if multiplier > eps:
        return float(residual)

    return float(max(residual, 0.0))


# ============================================================
# Continuous-time primal-dual dynamics
# ============================================================

def dynamics(
    t, yy, T, G, rho,
    line_coefficients,
    local_bounds,
    eps=0.0,
):
    """
    General nonquadratic multi-robot covering problem.

    Cost of agent v:
        f_v(z_v, z_-v) = rho_v ||z_v - T_v||^2
                        + sum_{u != v}
                        (
                            sqrt(1 + ||z_v-z_u||^2) - 1
                        ).

    The first term is target tracking.  The second term is a smooth
    nonquadratic proximity/communication cost.  Its gradient is globally
    Lipschitz.

    Shared group constraint:
        g^G(z) = -n_G^T
                (
                    (1 / |G|) sum_{u in G} z_u
                ) - c_G 
                <= 0.

    Therefore the corresponding feasible half-plane in the 2-D plot is
        n_G^T p + c_G >= 0.

    Individual constraint:
        h_v(z_v) = C_v z_v - d_v <= 0.

    Dynamics:
        dot z_v = - grad f_v - lambda_v grad g_v - C_v^T mu_v
        dot lambda_v = [g^G(z)]^+_{lambda_v}
        dot mu_v = [h_v(z_v)]^+_{mu_v}.
    """
    del t

    agents = _agent_order(T)
    idx = {v: i for i, v in enumerate(agents)}

    n = len(agents)
    q = 4

    lambda_start = 2 * n
    mu_start = 3 * n

    yy = np.asarray(yy, dtype=float)

    Z = yy[:2 * n].reshape(n, 2)

    z_dots = np.zeros((n, 2), dtype=float)
    lambda_dots = np.zeros(n, dtype=float)
    mu_dots = np.zeros((n, q), dtype=float)

    for v in agents:
        i = idx[v]

        z_v = Z[i]
        target_v = np.asarray(T[v][:2], dtype=float)

        # ----------------------------------------------------
        # Cost gradient
        # rho_v ||z_v-T_v||^2
        # -> 2 rho_v (z_v-T_v)
        # ----------------------------------------------------
        rho_v = _rho_value(rho, v, i)

        grad_f = 2.0 * rho_v * (z_v - target_v)

        # ----------------------------------------------------
        # Nonquadratic communication/proximity term
        # phi(d) = sqrt(1 + ||d||^2) - 1
        # grad phi(d) = d / sqrt(1 + ||d||^2)
        # ----------------------------------------------------
        for u in agents:
            if u == v:
                continue

            j = idx[u]
            diff = z_v - Z[j]

            grad_f += diff / np.sqrt(1.0 + float(diff @ diff))

        # ----------------------------------------------------
        # Shared group constraint
        # ----------------------------------------------------
        group = T[v][2]
        members = G[group]
        member_idx = [idx[u] for u in members]
        group_size = len(members)

        n_G = np.asarray(line_coefficients[group]["coef"], dtype=float)
        c_G = float(line_coefficients[group]["intercept"])

        group_center = Z[member_idx].mean(axis=0)

        # g^G(z) = -n_G^T center - c_G <= 0
        g_v = -float(n_G @ group_center) - c_G

        # grad_{z_v} g^G = -(1/|G|) n_G
        grad_g_v = -(1.0 / group_size) * n_G

        lambda_raw = float(yy[lambda_start + i])
        lambda_eff = max(lambda_raw, 0.0)

        # ----------------------------------------------------
        # Individual box constraint
        # ----------------------------------------------------
        C_v, d_v = local_C_d(local_bounds[v])
        h_v = C_v @ z_v - d_v

        mu_raw = np.asarray(
            yy[
                mu_start + q * i:
                mu_start + q * (i + 1)
            ],
            dtype=float,
        )
        mu_eff = np.maximum(mu_raw, 0.0)

        # ----------------------------------------------------
        # Primal dynamics
        # ----------------------------------------------------
        z_dots[i] = (
            -grad_f
            - lambda_eff * grad_g_v
            - C_v.T @ mu_eff
        )

        # ----------------------------------------------------
        # Shared multiplier dynamics
        # ----------------------------------------------------
        lambda_dots[i] = _projected_rate(
            g_v,
            lambda_raw,
            eps,
        )

        # ----------------------------------------------------
        # Individual multiplier dynamics
        # ----------------------------------------------------
        for j in range(q):
            mu_dots[i, j] = _projected_rate(
                h_v[j],
                mu_raw[j],
                eps,
            )

    return np.concatenate(
        [
            z_dots.reshape(-1),
            lambda_dots,
            mu_dots.reshape(-1),
        ]
    )


# ============================================================
# Initialization
# ============================================================

def make_init(
    T,
    *,
    seed: Optional[int] = None,
    init_mode: str = "around_targets",
    pos_scale: float = 0.15,
    lambda_scale: float = 4.0,
    mu_scale: float = 0.0,
    fixed_positions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Construct

        y0 = [z(0), Lambda(0), mu(0)].

    fixed_positions:
        If supplied, all experiments can use exactly the same z(0)
        while changing only the initial copied multipliers.
    """
    rng = np.random.default_rng(seed)

    agents = _agent_order(T)
    n, state_dim = _dim_counts(T)

    y0 = np.zeros(state_dim, dtype=float)

    # --------------------------------------------------------
    # Positions
    # --------------------------------------------------------
    if fixed_positions is not None:
        Z0 = np.asarray(fixed_positions, dtype=float).reshape(n, 2)

    elif init_mode == "zeros":
        Z0 = np.zeros((n, 2), dtype=float)

    elif init_mode == "around_targets":
        targets = np.asarray(
            [T[v][:2] for v in agents],
            dtype=float,
        )

        Z0 = targets + rng.normal(
            0.0,
            pos_scale,
            size=(n, 2),
        )

    else:
        raise ValueError(f"Unknown init_mode: {init_mode}")

    y0[:2 * n] = Z0.reshape(-1)

    # --------------------------------------------------------
    # Shared multipliers
    # --------------------------------------------------------
    lambda_start = _lambda_start_index(T)

    y0[
        lambda_start:
        lambda_start + n
    ] = rng.uniform(
        0.0,
        lambda_scale,
        size=n,
    )

    # --------------------------------------------------------
    # Individual multipliers
    # --------------------------------------------------------
    mu_start = _mu_start_index(T)

    if mu_scale > 0.0:
        y0[mu_start:] = rng.uniform(
            0.0,
            mu_scale,
            size=4 * n,
        )

    ensure_dual_nonneg_inplace(y0, T)

    return y0


# ============================================================
# Solver
# ============================================================

def solve(
    T, G, rho,
    line_coefficients,
    local_bounds,
    *,
    Tmax: float = 100.0,
    TimeStamp: int = 20_000,
    x_init: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    eps: float = 0.0,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    max_step: Optional[float] = None,
):
    """
    Solve the continuous-time primal-dual system.

    Returns
    -------
    t : ndarray, shape (TimeStamp,)
    X : ndarray, shape (TimeStamp, 7N)
    """
    if x_init is None:
        x_init = make_init(
            T, seed=seed,
        )

    y0 = np.asarray(x_init, dtype=float).reshape(-1).copy()

    _, expected_dim = _dim_counts(T)

    if y0.size != expected_dim:
        raise ValueError(
            f"Initial state has length {y0.size}, "
            f"but the general model requires {expected_dim}=7N entries."
        )

    ensure_dual_nonneg_inplace(y0, T)

    t_eval = np.linspace(
        0.0, Tmax, int(TimeStamp),
    )

    if max_step is None:
        max_step = Tmax / 2000.0

    sol = solve_ivp(
        fun=dynamics,
        t_span=(0.0, Tmax),
        y0=y0,
        t_eval=t_eval,
        args=(
            T, G, rho,
            line_coefficients,
            local_bounds, eps,
        ),
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        vectorized=False,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(
            f"solve_ivp failed: {sol.message}"
        )

    return sol.t, sol.y.T


# ============================================================
# State extraction
# ============================================================

def split_state(X, T):
    """
    Split a state or state trajectory into
        Z       : (..., N, 2)
        Lambda  : (..., N)
        Mu      : (..., N, 4)
    """
    X = np.asarray(X, dtype=float)

    n = len(T)

    Z = X[
        ..., :2 * n
    ].reshape(
        *X.shape[:-1], n, 2,
    )

    Lambda = X[
        ..., 2 * n: 3 * n
    ]

    Mu = X[
        ..., 3 * n:
    ].reshape(
        *X.shape[:-1],
        n, 4,
    )

    return Z, Lambda, Mu


# ============================================================
# Multiplier discrepancy Delta
# ============================================================

def compute_group_deltas(
    X, T, G,
):
    """
    For each physical shared constraint, choose the copied multiplier
    with the smallest INITIAL value as the reference.

    Returns
    -------
    Delta[(group, agent)] : trajectory lambda_v - lambda_reference
    references[group]     : reference agent
    """
    agents = _agent_order(T)
    idx = {v: i for i, v in enumerate(agents)}

    _, Lambda, _ = split_state(X, T)

    Delta = {}
    references = {}

    for group, members in G.items():
        member_idx = [idx[v] for v in members]

        initial_lambdas = Lambda[0, member_idx]
        ref_local_idx = int(np.argmin(initial_lambdas))

        ref_agent = members[ref_local_idx]
        ref_idx = idx[ref_agent]

        references[group] = ref_agent
        lambda_ref = Lambda[:, ref_idx]

        for v in members:
            i = idx[v]

            Delta[(group, v)] = (
                Lambda[:, i] - lambda_ref
            )

    return Delta, references


# ============================================================
# Exact ODE residual for numerical convergence checking
# ============================================================

def rhs_norm(
    t, X, T, G, rho,
    line_coefficients,
    local_bounds,
    eps=0.0,
):
    """
    Evaluate
        ||dot y(t)||

    from the implemented ODE rather than using numerical
    differentiation of X.
    """
    values = np.empty(len(t), dtype=float)

    for k, (tk, xk) in enumerate(zip(t, X)):
        values[k] = np.linalg.norm(
            dynamics(
                tk, xk, T, G, rho,
                line_coefficients,
                local_bounds,
                eps,
            )
        )

    return values