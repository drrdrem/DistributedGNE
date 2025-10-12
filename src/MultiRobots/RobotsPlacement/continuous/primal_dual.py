# -*- coding: utf-8 -*-
# Continuous-time distributed primal–dual (Algorithm 1) with local multipliers
# - ż_v = -∇_z f_v(z) - Σ_{c∈M_v} λ_{v,c} + box_term
# - λ̇_{v,c} = g_c(z)  with g_c(z) = Σ_{a∈Center[c]} z_a + b_c
# - μ̇ = [h(z)]_μ^+ for L∞ boxes around anchors A_v with radius r_v (optional)
#
# Cost per agent v (ρ on the FIRST term as requested):
#   f_v(z) = ρ_v ||z_v - A_v||^2 + Σ_{u∈N(v)} ||z_v - z_u||^2
#
# b_c is constructed from inputs: b_c = -|Center[c]| * C[c]
# If you want b_c = -C[c], set scale_b_by_group=False in solve().

from typing import Dict, Optional, Tuple, Iterable
import numpy as np
from scipy.integrate import solve_ivp


# -------------------- helpers --------------------
def neighbors_of(v, Agent, Center):
    neigh = set()
    for c in Agent[v]:
        neigh.update(Center[c])
    neigh.discard(v)
    return list(neigh)

def build_layout(Agent, use_box: bool):
    """
    y layout:
      if use_box: per agent k=6: [x,y, μx+,μx-, μy+,μy-] else k=2: [x,y]
      Lambdas are 2D (x,y) per (v,c): lam_idx[(v,c,'x')], lam_idx[(v,c,'y')]
    """
    agents = list(Agent.keys())
    n = len(agents)
    k = 6 if use_box else 2
    lam_base = k * n
    lam_idx, off = {}, lam_base
    for v in agents:
        for c in Agent[v]:
            lam_idx[(v, c, 'x')] = off
            lam_idx[(v, c, 'y')] = off + 1
            off += 2
    total_len = off
    return k, lam_base, lam_idx, total_len

def make_b(Center, C, *, scale_b_by_group=True):
    """
    For equality: ψ_c(z) = Σ z_a + b_c.
      - If scale_b_by_group=True: b_c = -|G| * C[c]  (so ∂ψ/∂z_v = I)
      - Else:                     b_c = - C[c]       (then ∂ψ/∂z_v = I/|G|; adjust below)
    """
    b = {}
    for c, members in Center.items():
        if scale_b_by_group:
            b[c] = -len(members) * np.array(C[c], dtype=float)
        else:
            b[c] = -np.array(C[c], dtype=float)
    return b

def mu_indices(Agent, r):
    if r is None: return []
    k = 6
    idxs = []
    for i, _ in enumerate(Agent):
        base = k * i
        idxs += [base+2, base+3, base+4, base+5]
    return idxs

def ensure_mu_nonneg_inplace(y, Agent, r):
    for j in mu_indices(Agent, r):
        if y[j] < 0: y[j] = np.abs(y[j])

# -------------------- dynamics --------------------
import numpy as np

def dynamics(t, y, Agent, Center, C, rho, A, r=None, *, scale_b_by_group=True):
    """
    ż_v = -∇f_v(z) - Σ_{c∈Agent[v]} λ_{v,c} * G_{v,c} + box_term
    λ̇_{v,c} = ψ_c(z)         (equality; no projection)
    μ̇       = [h(z)]_μ^+     (L∞ boxes; projected)

    Here ψ_c(z) = Σ_{a∈Center[c]} z_a + b_c  (vector in R^2).
    G_{v,c} = I if scale_b_by_group=True; else G_{v,c} = I/|Center[c]|.
    """
    use_box = r is not None
    agents = list(Agent.keys())
    n = len(agents)
    k, lam_base, lam_idx, total_len = build_layout(Agent, use_box)
    b = make_b(Center, C, scale_b_by_group=scale_b_by_group)

    def z_of(i):  # i: agent index in 'agents'
        return np.array([y[k*i], y[k*i+1]], dtype=float)

    f = np.zeros(total_len, dtype=float)

    # ---- primal & (optional) μ dynamics ----
    for i, v in enumerate(agents):
        z_v = z_of(i)
        # ∇ f_v with ρ on FIRST term
        Nv = neighbors_of(v, Agent, Center)
        deg = len(Nv)
        grad = 2.0 * rho[v] * (z_v - np.array(A[v], dtype=float))
        if deg > 0:
            sum_u = np.sum([z_of(agents.index(u)) for u in Nv], axis=0)
            grad += 2.0 * (deg * z_v - sum_u)

        # - Σ λ_{v,c} * G_{v,c}
        lam_term = np.zeros(2, dtype=float)
        for c in Agent[v]:
            lam_vc = np.array([y[lam_idx[(v, c, 'x')]], y[lam_idx[(v, c, 'y')]]], dtype=float)
            if scale_b_by_group:
                G_vc = np.eye(2)               # since ψ uses SUM
            else:
                G_vc = np.eye(2) / len(Center[c])  # since ψ uses MEAN
            lam_term += G_vc @ lam_vc

        # box term
        box_term = np.zeros(2, dtype=float)
        if use_box:
            mu_xp, mu_xm = y[k*i+2], y[k*i+3]
            mu_yp, mu_ym = y[k*i+4], y[k*i+5]
            box_term = np.array([-mu_xp + mu_xm, -mu_yp + mu_ym], dtype=float)

        dz = -grad - lam_term + box_term
        f[k*i]   = dz[0]
        f[k*i+1] = dz[1]

        # μ̇ = [h]_μ^+
        if use_box:
            ax, ay, rv = A[v][0], A[v][1], r[v]
            h_xp = z_v[0] - (ax + rv)
            h_xm = -(z_v[0] - (ax - rv))
            h_yp = z_v[1] - (ay + rv)
            h_ym = -(z_v[1] - (ay - rv))
            f[k*i+2] = (h_xp > 0.0 or y[k*i+2] > 0.0) * h_xp
            f[k*i+3] = (h_xm > 0.0 or y[k*i+3] > 0.0) * h_xm
            f[k*i+4] = (h_yp > 0.0 or y[k*i+4] > 0.0) * h_yp
            f[k*i+5] = (h_ym > 0.0 or y[k*i+5] > 0.0) * h_ym

    # ---- dual λ dynamics for each local copy (vector in R^2) ----
    for v in agents:
        for c in Agent[v]:
            members = Center[c]
            sum_z = np.sum([z_of(agents.index(a)) for a in members], axis=0)
            g = sum_z + b[c]  # ψ_c(z) (R^2)
            f[lam_idx[(v, c, 'x')]] = g[0]
            f[lam_idx[(v, c, 'y')]] = g[1]

    return f

# -------------------- Solver --------------------
def make_init(Agent: Dict, r: Optional[Dict] = None,
              low: float = -3.0, high: float = 1.0,
              seed: Optional[int] = None) -> np.ndarray:
    """Random init with μ ≥ 0 (λ unrestricted)."""
    if seed is not None:
        np.random.seed(seed)
    use_box = r is not None
    k = 6 if use_box else 2
    n = len(Agent)
    # 2 λ slots per (v,c)
    num_lams = sum(2 * len(Agent[v]) for v in Agent)
    L = k * n + num_lams
    y0 = low + (high - low) * np.random.rand(L)
    ensure_mu_nonneg_inplace(y0, Agent, r)
    return y0

def solve(Agent: Dict, Center: Dict, C: Dict, rho: Dict, A: Dict,
          r: Optional[Dict] = None, *,
          Tmax: float = 50.0, TimeStamp: int = 50_000,
          x_init: Optional[np.ndarray] = None, seed: Optional[int] = None,
          psi_mode: str = "sum",
          rtol: float = 1e-6, atol: float = 1e-8, max_step: Optional[float] = None
          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper around solve_ivp. Returns (t_eval, X) with X shape (T, dim).
    """
    if x_init is None:
        x_init = make_init(Agent, r=r, seed=seed)
    y0 = np.asarray(x_init, dtype=float).reshape(-1)
    ensure_mu_nonneg_inplace(y0, Agent, r)

    t_eval = np.linspace(0.0, Tmax, int(TimeStamp))
    if max_step is None:
        max_step = Tmax / (20 * TimeStamp) * 100.0  # finite but permissive

    sol = solve_ivp(
        fun=dynamics,
        t_span=(0.0, Tmax),
        y0=y0,
        t_eval=t_eval,
        args=(Agent, Center, C, rho, A, r),
        rtol=rtol, atol=atol, max_step=max_step, vectorized=False
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    return sol.t, sol.y.T