import numpy as np

# ---------- helpers ----------
def neighbors_of(v, Agent, Center):
    """Agents that share at least one constraint with v (no duplicates, no self)."""
    neigh = set()
    for c in Agent[v]:
        neigh.update(Center[c])
    neigh.discard(v)
    return list(neigh)

def build_layout_adaptive_with_z(Agent, use_box: bool):
    """
    Per-agent block (K scalars):
      if use_box: [ x, y, mu_x+, mu_x-, mu_y+, mu_y-,  k,  z_x, z_y ]  -> K = 9
      else:       [ x, y,                                   k,  z_x, z_y ]  -> K = 5
    Then all local lambdas λ_{v,c}^x, λ_{v,c}^y stacked (for c in Agent[v], v in Agent).
    Returns: (K, lam_base, lam_idx, total_len)
    """
    agents = list(Agent.keys())
    n = len(agents)
    K = 9 if use_box else 5
    lam_base = K * n
    lam_idx, off = {}, lam_base
    for v in agents:
        for c in Agent[v]:
            lam_idx[(v, c, 'x')] = off
            lam_idx[(v, c, 'y')] = off + 1
            off += 2
    return K, lam_base, lam_idx, off

def idx_maps(K, use_box):
    """Index helpers for a given K/use_box."""
    def ix(v): return K*v
    def iy(v): return K*v + 1
    if use_box:
        def imxp(v): return K*v + 2
        def imxm(v): return K*v + 3
        def imyp(v): return K*v + 4
        def imym(v): return K*v + 5
        def ik(v):   return K*v + 6
        def izx(v):  return K*v + 7
        def izy(v):  return K*v + 8
        return ix, iy, imxp, imxm, imyp, imym, ik, izx, izy
    else:
        def ik(v):   return K*v + 2
        def izx(v):  return K*v + 3
        def izy(v):  return K*v + 4
        return ix, iy, None, None, None, None, ik, izx, izy

# ---------- continuous-time RHS (Adaptive gains with local g_i and b_i) ----------
def dynamics_adaptive_gi(
    t, y,
    Agent, Center, C, rho, A,
    r=None,         # dict r[v] for L∞ boxes (optional)
    gamma=None      # dict gamma[v] > 0 for k̇_v = gamma[v] * ||rho^v||^2 (default 1.0)
):
    """
    Adaptive-gain continuous-time primal-dual with consensus filter z_i and local g_i(x_i).

    Objective:  f_v(x) = ρ_v||x_v - A_v||^2 + Σ_{u∈N(v)} ||x_v - x_u||^2
    Local residual:  g_i(x_i) = Σ_{c∈Agent[v]} ( A_i x_v + b_i^c ),  with A_i=I,  b_i^c = -C[c]
    Consensus filter:  ż_v = Σ_{j∈N(v)} (λ_v^tot - λ_j^tot)
    Dual:  λ̇_v^tot = g_i(x_v) - z_v - Σ_{j∈N(v)} (λ_v^tot - λ_j^tot)
           (applied identically to each local copy λ_{v,c})
    No projection on λ.

    Boxes (if r is given): μ dynamics via projected [h]_μ^+ and box force in ẋ.

    Adaptive gain:
      ρ^v := Σ_{j∈N(v)} (x_v - x_j)
      k̇_v = γ_v ||ρ^v||^2
      ẋ_v = -∇f_v(x) - λ_v^tot - Σ_{j∈N(v)} (k_v - k_j)(x_v - x_j) + box_term
    """
    use_box = r is not None
    agents = list(Agent.keys())
    n = len(agents)
    if gamma is None:
        gamma = {v: 1.0 for v in agents}

    # layout & indices
    K, lam_base, lam_idx, total_len = build_layout_adaptive_with_z(Agent, use_box)
    ix, iy, imxp, imxm, imyp, imym, ik, izx, izy = idx_maps(K, use_box)

    # small accessors
    def xvec(v): return np.array([y[ix(v)], y[iy(v)]])
    def zvec(v): return np.array([y[izx(v)], y[izy(v)]])

    dy = np.zeros(total_len, dtype=float)

    # --- Precompute aggregated lambdas per agent: λ_v^tot = Σ_c λ_{v,c} ---
    lam_tot = {v: np.zeros(2) for v in agents}
    for v in agents:
        for c in Agent[v]:
            lam_tot[v] += np.array([y[lam_idx[(v,c,'x')]], y[lam_idx[(v,c,'y')]]])

    # --- x, μ, k, z dynamics ---
    for v in agents:
        x_v = xvec(v)
        N_v = neighbors_of(v, Agent, Center)

        # ∇f_v(x)
        grad = 2*rho[v]*(x_v - np.array(A[v], dtype=float))
        if len(N_v) > 0:
            sum_u = np.sum([xvec(u) for u in N_v], axis=0)
            grad += 2*(len(N_v)*x_v - sum_u)

        # local dual force: -λ_v^tot
        lam_force = -lam_tot[v]

        # adaptive consensus term:  -Σ_j (k_v - k_j)(x_v - x_j)
        k_v = y[ik(v)]
        adapt = np.zeros(2)
        for j in N_v:
            k_j = y[ik(j)]
            adapt += (k_v - k_j) * (x_v - xvec(j))
        adapt = -adapt

        # box term via μ (if used)
        box_term = np.zeros(2)
        if use_box:
            mu_xp, mu_xm = y[imxp(v)], y[imxm(v)]
            mu_yp, mu_ym = y[imyp(v)], y[imym(v)]
            box_term = np.array([-mu_xp + mu_xm, -mu_yp + mu_ym])

        xdot = -grad + lam_force + adapt + box_term
        dy[ix(v)] = xdot[0]
        dy[iy(v)] = xdot[1]

        # μ̇ = [h]_μ^+ (projected dynamics for L∞ box)
        if use_box:
            ax, ay, rv = A[v][0], A[v][1], r[v]
            hxp = x_v[0] - (ax + rv)          # ≤ 0
            hxm = -(x_v[0] - (ax - rv))       # ≤ 0
            hyp = x_v[1] - (ay + rv)          # ≤ 0
            hym = -(x_v[1] - (ay - rv))       # ≤ 0
            dy[imxp(v)] = (hxp > 0.0 or y[imxp(v)] > 0.0) * hxp
            dy[imxm(v)] = (hxm > 0.0 or y[imxm(v)] > 0.0) * hxm
            dy[imyp(v)] = (hyp > 0.0 or y[imyp(v)] > 0.0) * hyp
            dy[imym(v)] = (hym > 0.0 or y[imym(v)] > 0.0) * hym

        # ρ^v and k̇_v
        rho_v = np.sum([x_v - xvec(j) for j in N_v], axis=0) if len(N_v) > 0 else np.zeros(2)
        dy[ik(v)] = gamma[v] * float(np.dot(rho_v, rho_v))

        # ż_v = Σ_j (λ_v^tot - λ_j^tot)
        zdot = np.zeros(2)
        for j in N_v:
            zdot += (lam_tot[v] - lam_tot[j])
        dy[izx(v)] = zdot[0]
        dy[izy(v)] = zdot[1]

    # --- λ̇: local g_i(x_i) - z_i - Σ_j (λ_i - λ_j)  (no projection) ---
    # Build local g_i(x_i) with A_i=I and b_i^c=-C[c]; sum over c∈Agent[v]
    for v in agents:
        x_v = xvec(v)
        N_v = neighbors_of(v, Agent, Center)

        b_sum = np.zeros(2)
        for c in Agent[v]:
            b_sum += -np.array(C[c], dtype=float)   # b_i^c = -C[c]
        g_i = len(Agent[v]) * x_v + b_sum           # Σ_c (I x_v + b_i^c)

        # consensus term on λ_tot
        cons = np.zeros(2)
        for j in N_v:
            cons += (lam_tot[v] - lam_tot[j])

        lam_tot_dot = g_i - zvec(v) - cons

        # apply same λ̇_i^tot to each local copy (v,c)
        for c in Agent[v]:
            dy[lam_idx[(v,c,'x')]] = lam_tot_dot[0]
            dy[lam_idx[(v,c,'y')]] = lam_tot_dot[1]

    return dy