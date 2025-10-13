import numpy as np
from typing import Dict, List, Tuple, Optional, Iterable
from scipy.integrate import solve_ivp

def dynamics(t, yy, T, G, rho, line_coefficients, eps=0.0):
    """
    Continuous-time dynamics for each agent v:
        ż_v = -∇_{z_v} L_v(z, λ_v) = -∇ f_v(z) - λ_v ∇ g^G_v(z)
        λ̇_v = [ g^G(z) ]^+_{λ_v}
    where:
        f_v(z) = ρ_v ||z_v - T_v||^2 + Σ_{u≠v} ||z_v - z_u||^2
        g^G(z) = -n_G^T ( (1/|G|) Σ_{u∈G} z_u ) - c_G

    Args:
        t : time (unused; kept for ODE signature)
        yy: flat state vector [x0,y0, x1,y1, ..., λ0, λ1, ...]
        T : dict {v: (T_x, T_y, group_id)}
        G : dict {group_id: [agent ids in that group]}
        rho : dict {v: ρ_v >= 0}
        line_coefficients : dict {group_id: {'coef': [n_x, n_y], 'intercept': c}}
        eps : small deadzone to reduce λ projection chattering

    Returns:
        list of derivatives in the same order as yy
    """
    # Stable ordering that works even if agent ids are not 0..n-1
    agents = list(T.keys())
    idx = {v: i for i, v in enumerate(agents)}
    n = len(agents)

    # Split state into positions and multipliers (assume order [x0,y0,..., λ0,λ1,...])
    la_start = 2 * n
    Z = np.empty((n, 2), dtype=float)
    for v, i in idx.items():
        Z[i, 0] = yy[2 * i]
        Z[i, 1] = yy[2 * i + 1]

    # Precompute Σ z_u to get Σ_{u≠v} cheaply
    total = Z.sum(axis=0)

    z_dots = []
    lam_dots = []

    for v in agents:
        i = idx[v]
        z_i = Z[i]
        T_i = np.array(T[v][:2], dtype=float)

        # ---- cost gradient (ρ_v ON THE FIRST TERM) ----
        # ∇ f_v = 2 ρ_v (z_i - T_i) + 2[ (n-1) z_i - Σ_{u≠i} z_u ]
        sum_others = total - z_i
        grad_f = 2.0 * rho[v] * (z_i - T_i) + 2.0 * ((n - 1) * z_i - sum_others)

        # ---- group constraint ----
        group = T[v][2]
        members = G[group]
        m_idx = [idx[u] for u in members]
        m = len(members)

        nG = np.array(line_coefficients[group]['coef'], dtype=float)  # normal n_G
        cG = float(line_coefficients[group]['intercept'])             # intercept c_G

        avg = Z[m_idx].mean(axis=0)          # (1/|G|) Σ_{u∈G} z_u
        g = -float(nG @ avg) - cG            # g^G(z)

        # ∂g/∂z_v = -(1/|G|) n_G  (only if v in this group)
        grad_g_v = -(1.0 / m) * nG

        lam_i = float(yy[la_start + i])

        # ---- primal and dual dynamics ----
        # ż_v = -∇ f_v - λ_v ∇ g_v  (note grad_g_v is negative)
        z_dot = -grad_f - lam_i * grad_g_v

        # λ̇_v = g if (λ_v > 0 or g > 0) else 0
        if (lam_i > 0.0 + eps) or (g > 0.0 + eps):
            lam_dot = g
        else:
            lam_dot = 0.0

        z_dots.extend(z_dot.tolist())
        lam_dots.append(lam_dot)

    return (np.concatenate([np.asarray(z_dots), np.asarray(lam_dots)])).tolist()


# ------------------------------------------------------------
# Helpers for state layout: [x0,y0, x1,y1, ..., λ0, λ1, ...]
# ------------------------------------------------------------
def _agent_order(T: Dict[int, Iterable[float]]) -> List[int]:
    """Stable agent ordering (keys may not be 0..n-1)."""
    return list(T.keys())

def _dim_counts(T: Dict[int, Iterable[float]]) -> Tuple[int, int]:
    """Return (#agents, state_dim). Here state_dim = 3n (2n for z, n for λ)."""
    n = len(T)
    return n, 3 * n

def _lambda_start_index(T: Dict[int, Iterable[float]]) -> int:
    n, _ = _dim_counts(T)
    return 2 * n

def ensure_lambda_nonneg_inplace(y0: np.ndarray, T: Dict[int, Iterable[float]]) -> None:
    """Project λ components in y0 to the nonnegative orthant, in place."""
    la_start = _lambda_start_index(T)
    y0[la_start:] = np.maximum(y0[la_start:], 0.0)

def make_init(
    T: Dict[int, Iterable[float]],
    *,
    seed: Optional[int] = None,
    init_mode: str = "around_targets",
    pos_scale: float = 0.05,
    lam_scale: float = 0.1,
) -> np.ndarray:
    """
    Build an initial state consistent with the layout [z, λ].
      init_mode:
        - "zeros": z = 0, λ = 0
        - "around_targets": z = T_v + small noise, λ >= 0 small
        - "random_box": z uniform in a loose box inferred from T, λ >= 0 small
    """
    rng = np.random.default_rng(seed)
    agents = _agent_order(T)
    n, L = _dim_counts(T)
    y0 = np.zeros(L, dtype=float)

    # Positions
    if init_mode == "zeros":
        pass
    elif init_mode == "around_targets":
        all_T = np.asarray([T[v][:2] for v in agents], dtype=float)  # (n, 2)
        noise = rng.normal(0.0, pos_scale, size=(n, 2))
        Z0 = all_T + noise
        y0[: 2 * n] = Z0.reshape(-1)
    elif init_mode == "random_box":
        all_T = np.asarray([T[v][:2] for v in agents], dtype=float)
        lo = all_T.min(axis=0) - 1.0
        hi = all_T.max(axis=0) + 1.0
        Z0 = rng.uniform(lo, hi, size=(n, 2))
        y0[: 2 * n] = Z0.reshape(-1)
    else:
        raise ValueError(f"Unknown init_mode: {init_mode}")

    # Lambdas: small nonnegative
    la_start = _lambda_start_index(T)
    y0[la_start:] = np.abs(rng.normal(0.0, lam_scale, size=n))
    return y0

# ------------------------------------------------------------
# SOLVER WRAPPER (mirrors your signature and behavior)
# ------------------------------------------------------------
def solve(
    T: Dict[int, Iterable[float]],
    G: Dict[int, List[int]],
    rho: Dict[int, float],
    line_coefficients: Dict[int, Dict[str, Iterable[float]]],
    *,
    Tmax: float = 50.0,
    TimeStamp: int = 50_000,
    x_init: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    eps: float = 0.0,                 # passthrough to dynamics
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_step: Optional[float] = None,
    dynamics_fn=None,                 # allow swapping in a different dynamics
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper around solve_ivp for your continuous-time primal–dual flow.
    Returns:
        t_eval : (T,) time grid
        X      : (T, dim) states in the same order as yy = [x0,y0,..., λ0,λ1,...]
    """
    if dynamics_fn is None:
        # By default, we expect `dynamics` to be present in the module scope
        dynamics_fn = dynamics

    if x_init is None:
        x_init = make_init(T, seed=seed, init_mode="around_targets")

    y0 = np.asarray(x_init, dtype=float).reshape(-1)
    ensure_lambda_nonneg_inplace(y0, T)

    t_eval = np.linspace(0.0, Tmax, int(TimeStamp))
    if max_step is None:
        # permissive but finite (helps some stiff-ish regimes)
        max_step = Tmax / (20 * TimeStamp) * 100.0

    sol = solve_ivp(
        fun=dynamics_fn,
        t_span=(0.0, Tmax),
        y0=y0,
        t_eval=t_eval,
        args=(T, G, rho, line_coefficients, eps),
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        vectorized=False,
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    return sol.t, sol.y.T  # shape (T, dim)