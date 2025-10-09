# -*- coding: utf-8 -*-
# Continuous-time distributed primal–dual (Algorithm 1) with local multipliers
# - ż_v = -∇_z f_v(z) - Σ_{c∈M_v} λ_{v,c} + box_term
# - λ̇_{v,c} = g_c(z)  with g_c(z) = Σ_{a∈Center[c]} z_a + b_c
# - μ̇ = [h(z)]_μ^+ for L∞ boxes around anchors A_v with radius r_v (optional)
#
# b_c is constructed from inputs: b_c = -|Center[c]| * C[c]
# If you want b_c = -C[c], set scale_b_by_group=False in solve().

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---------- helpers ----------
def neighbors_of(v, Agent, Center):
    """Agents that share at least one constraint with v (de-duplicated, no self)."""
    neigh = set()
    for c in Agent[v]:
        neigh.update(Center[c])
    neigh.discard(v)
    return list(neigh)

def build_layout(Agent, use_box: bool):
    """
    Returns (k, lam_base, lam_idx, total_len).
    Layout if use_box (k=6 per agent):
        y = [ x0,y0, μx+,μx-, μy+,μy-,  x1,y1, μ...,  ...,  lambdas... ]
    else (k=2 per agent):
        y = [ x0,y0,  x1,y1,  ...,  lambdas... ]
    Lambdas are 2D per (v,c) stored in order: for v in Agent: for c in Agent[v]: (x,y)
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
    Construct b_c for each constraint c:
      if scale_b_by_group: b_c = -|Center[c]| * C[c]
      else:                b_c = -C[c]
    """
    b = {}
    for c, members in Center.items():
        if scale_b_by_group:
            b[c] = -len(members) * np.array(C[c], dtype=float)
        else:
            b[c] = -np.array(C[c], dtype=float)
    return b

# ---------- continuous-time RHS (Algorithm 1) ----------
def dynamics(t, y, Agent, Center, C, rho, A, r=None, *, scale_b_by_group=True):
    """
    Algorithm 1 (continuous-time distributed primal-dual), 2D positions per agent.
      f_v(z) = ρ_v||z_v - A_v||^2 + Σ_{u∈N(v)} ||z_v - z_u||^2
      g_c(z) = Σ_{a∈Center[c]} z_a + b_c,  with b_c from make_b(...)
      Box constraints per agent v: |z_v - A_v|_∞ ≤ r_v via μ = (μx+, μx-, μy+, μy-) ≥ 0

    Dynamics:
      ż_v = -∇f_v(z) - Σ_{c∈Agent[v]} λ_{v,c} + [ -μx+ + μx-,  -μy+ + μy- ]
      λ̇_{v,c} = g_c(z)        (same residual for every local copy on c)
      μ̇ = [h(z)]_μ^+          with h_x+ = x - (A_x + r), h_x- = -(x - (A_x - r)), etc.
    """
    use_box = r is not None
    agents = list(Agent.keys())
    n = len(agents)
    k, lam_base, lam_idx, total_len = build_layout(Agent, use_box)
    b = make_b(Center, C, scale_b_by_group=scale_b_by_group)

    def z_of(v): return np.array([y[k*v], y[k*v+1]])

    f = np.zeros(total_len, dtype=float)

    # ż and (optionally) μ̇
    for v in agents:
        z_v = z_of(v)
        # ∇f_v
        Nv = neighbors_of(v, Agent, Center)
        deg = len(Nv)
        grad = 2 * rho[v] * (z_v - np.array(A[v], dtype=float))
        if deg > 0:
            sum_u = np.sum([z_of(u) for u in Nv], axis=0)
            grad += 2 * (deg * z_v - sum_u)

        # -Σ local λ_{v,c}
        lam_sum = np.zeros(2)
        for c in Agent[v]:
            lam_sum += np.array([y[lam_idx[(v, c, 'x')]], y[lam_idx[(v, c, 'y')]]])

        # box term (if any): [-μx+ + μx-, -μy+ + μy-]
        box_term = np.zeros(2)
        if use_box:
            mu_xp, mu_xm = y[k*v+2], y[k*v+3]
            mu_yp, mu_ym = y[k*v+4], y[k*v+5]
            box_term = np.array([-mu_xp + mu_xm, -mu_yp + mu_ym])

        dz = -grad - lam_sum + box_term
        f[k*v]   = dz[0]
        f[k*v+1] = dz[1]

        # μ̇ = [h]_μ^+   (only if boxes are active)
        if use_box:
            ax, ay, rv = A[v][0], A[v][1], r[v]
            h_xp = z_v[0] - (ax + rv)           # ≤ 0
            h_xm = -(z_v[0] - (ax - rv))        # ≤ 0
            h_yp = z_v[1] - (ay + rv)           # ≤ 0
            h_ym = -(z_v[1] - (ay - rv))        # ≤ 0

            mu_xp, mu_xm = y[k*v+2], y[k*v+3]
            mu_yp, mu_ym = y[k*v+4], y[k*v+5]

            f[k*v+2] = (h_xp > 0.0 or mu_xp > 0.0) * h_xp
            f[k*v+3] = (h_xm > 0.0 or mu_xm > 0.0) * h_xm
            f[k*v+4] = (h_yp > 0.0 or mu_yp > 0.0) * h_yp
            f[k*v+5] = (h_ym > 0.0 or mu_ym > 0.0) * h_ym

    # λ̇_{v,c} = g_c(z) for each local copy (same g for all agents on c)
    for v in agents:
        for c in Agent[v]:
            members = Center[c]
            sum_z = np.sum([z_of(a) for a in members], axis=0)
            g = sum_z + b[c]
            f[lam_idx[(v, c, 'x')]] = g[0]
            f[lam_idx[(v, c, 'y')]] = g[1]

    return f