import numpy as np

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