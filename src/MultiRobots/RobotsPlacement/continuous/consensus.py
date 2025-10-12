# adaptive_eq.py
import numpy as np
from typing import Dict, Optional, Tuple
from scipy.integrate import solve_ivp

# ---------- helpers ----------
def neighbors_of(v, Agent: Dict, Center: Dict):
    neigh = set()
    for c in Agent[v]:
        neigh.update(Center[c])
    neigh.discard(v)
    return list(neigh)

def block_size() -> int:
    """Per-agent block (no μ): [x, y, k, z_x, z_y] -> K = 5"""
    return 5

def idx_maps(K=5):
    def ix(v):  return K*v
    def iy(v):  return K*v + 1
    def ik(v):  return K*v + 2
    def izx(v): return K*v + 3
    def izy(v): return K*v + 4
    return ix, iy, ik, izx, izy

def build_layout_with_z(Agent: Dict):
    """
    State: [for v: x,y,k,zx,zy], then all local lambdas λ_{v,c}^x, λ_{v,c}^y
    Returns: (K, lam_base, lam_idx, total_len)
    """
    agents = list(Agent.keys())
    n = len(agents)
    K = block_size()
    lam_base = K * n
    lam_idx, off = {}, lam_base
    for v in agents:
        for c in Agent[v]:
            lam_idx[(v, c, 'x')] = off
            lam_idx[(v, c, 'y')] = off + 1
            off += 2
    return K, lam_base, lam_idx, off

# ---------- projection onto L∞ box (directional) ----------
def project_direction_box(x: np.ndarray, u: np.ndarray, A: np.ndarray, r: float, eps: float=0.0) -> np.ndarray:
    """
    Project direction u so that ẋ = u does not leave Ω = { |x-A|_∞ ≤ r }.
    If x at upper bound in a coord and u pushes outward, zero that coord; similarly for lower bound.
    """
    out = u.copy()
    # x-dimension
    ubx, lbx = A[0] + r, A[0] - r
    if x[0] >= ubx - eps and out[0] > 0: out[0] = 0.0
    if x[0] <= lbx + eps and out[0] < 0: out[0] = 0.0
    # y-dimension
    uby, lby = A[1] + r, A[1] - r
    if x[1] >= uby - eps and out[1] > 0: out[1] = 0.0
    if x[1] <= lby + eps and out[1] < 0: out[1] = 0.0
    return out

# ---------- dynamics (Algorithm 2, equality; NO μ, NO λ-projection) ----------
def dynamics(
    t, y,
    Agent: Dict, Center: Dict, C: Dict, rho: Dict, A: Dict,
    r: Optional[Dict]=None,         # if None: unconstrained; else L∞ box per agent
    gamma: Optional[Dict]=None      # k̇_v = γ_v ||ρ^v||^2
):
    agents = list(Agent.keys())
    n = len(agents)
    if gamma is None:
        gamma = {v: 1.0 for v in agents}

    K, lam_base, lam_idx, total_len = build_layout_with_z(Agent)
    ix, iy, ik, izx, izy = idx_maps(K)

    def xvec(v): return np.array([y[ix(v)], y[iy(v)]])
    def zvec(v): return np.array([y[izx(v)], y[izy(v)]])

    dy = np.zeros(total_len, dtype=float)

    # Precompute λ_v^tot = Σ_c λ_{v,c}
    lam_tot = {v: np.zeros(2) for v in agents}
    for v in agents:
        for c in Agent[v]:
            lam_tot[v] += np.array([y[lam_idx[(v,c,'x')]], y[lam_idx[(v,c,'y')]]])

    # ----- primal x, adaptive k, consensus z -----
    for v in agents:
        xv = xvec(v)
        Nv = neighbors_of(v, Agent, Center)

        # ∇ J_v(x) = 2 ρ_v (x_v - A_v) + 2( deg*x_v - Σ_{u∈N(v)} x_u )
        grad = 2.0 * rho[v] * (xv - np.array(A[v], dtype=float))
        if len(Nv) > 0:
            sum_u = np.sum([xvec(u) for u in Nv], axis=0)
            grad += 2.0 * (len(Nv) * xv - sum_u)

        # adaptive term  - Σ_j (k_v - k_j)(x_v - x_j)
        kv = y[ik(v)]
        adapt = np.zeros(2)
        for j in Nv:
            kj = y[ik(j)]
            adapt += (kv - kj) * (xv - xvec(j))
        adapt = -adapt

        # dual force  - λ_v^tot   (since ∂g_v/∂x_v = I when g_v(x_v)=x_v + b)
        dual_force = -lam_tot[v]

        # raw direction before projection
        u = -grad + dual_force + adapt

        # projection onto Ω_v if r is given
        if r is not None:
            u = project_direction_box(xv, u, np.array(A[v], dtype=float), float(r[v]))

        dy[ix(v)] = u[0]
        dy[iy(v)] = u[1]

        # ρ^v and k̇_v
        rho_v = np.sum([xv - xvec(j) for j in Nv], axis=0) if len(Nv) > 0 else np.zeros(2)
        dy[ik(v)] = gamma[v] * float(rho_v @ rho_v)

        # ż_v = Σ_j (λ_v^tot - λ_j^tot)
        zdot = np.zeros(2)
        for j in Nv:
            zdot += (lam_tot[v] - lam_tot[j])
        dy[izx(v)] = zdot[0]
        dy[izy(v)] = zdot[1]

    # ----- λ̇: g_v(x_v) - z_v - Σ_j (λ_v - λ_j)  (NO projection; equality setting) -----
    for v in agents:
        xv = xvec(v)
        Nv = neighbors_of(v, Agent, Center)

        # local residual g_v(x_v) = Σ_{c∈Agent[v]} (I*x_v + b_v^c)
        # with b_v^c = -C[c] (your choice)
        b_sum = np.zeros(2)
        for c in Agent[v]:
            b_sum += -np.array(C[c], dtype=float)
        g_v = len(Agent[v]) * xv + b_sum

        consensus = np.zeros(2)
        for j in Nv:
            consensus += (lam_tot[v] - lam_tot[j])

        lam_tot_dot = g_v - zvec(v) - consensus

        # apply same derivative to each local copy (v,c)
        for c in Agent[v]:
            dy[lam_idx[(v,c,'x')]] = lam_tot_dot[0]
            dy[lam_idx[(v,c,'y')]] = lam_tot_dot[1]

    return dy


# from adaptive_eq import dynamics_adaptive_eq, build_layout_with_z, block_size

def total_state_len(Agent: Dict) -> int:
    K = block_size()
    n = len(Agent)
    num_lams = sum(2 * len(Agent[v]) for v in Agent)
    return K * n + num_lams

def make_init_eq(Agent: Dict, low=-3.0, high=1.0, seed: Optional[int]=None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    L = total_state_len(Agent)
    return low + (high - low) * np.random.rand(L)

def solve_adaptive_eq(
    Agent: Dict, Center: Dict, C: Dict, rho: Dict, A: Dict,
    r: Optional[Dict]=None, gamma: Optional[Dict]=None,
    Tmax: float=100.0, TimeStamp: int=100_000,
    x_init: Optional[np.ndarray]=None, seed: Optional[int]=None,
    rtol: float=1e-6, atol: float=1e-8, max_step: Optional[float]=None
) -> Tuple[np.ndarray, np.ndarray]:
    if x_init is None:
        x_init = make_init_eq(Agent, seed=seed)

    y0 = np.asarray(x_init, dtype=float).reshape(-1)

    t_eval = np.linspace(0.0, Tmax, int(TimeStamp))
    if max_step is None:
        max_step = Tmax / (20 * TimeStamp) * 100.0

    sol = solve_ivp(
        fun=dynamics,
        t_span=(0.0, Tmax),
        y0=y0,
        t_eval=t_eval,
        args=(Agent, Center, C, rho, A, r, gamma),
        rtol=rtol, atol=atol, max_step=max_step, vectorized=False
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    return sol.t, sol.y.T