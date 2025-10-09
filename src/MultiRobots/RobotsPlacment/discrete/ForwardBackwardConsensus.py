import numpy as np

def _neighbors_for_agent(v, Agent, Center):
    neigh = set()
    for c in Agent[v]:
        neigh.update(Center[c])
    neigh.discard(v)
    return list(neigh)

def _make_indices(Agent):
    agents = list(Agent.keys())
    n = len(agents)
    pos_idx = {v: (2*i, 2*i+1) for i, v in enumerate(agents)}  # x and z are 2D
    x_base, z_base, lam_base = 0, 2*n, 4*n
    lam_idx, off = {}, lam_base
    for v in agents:
        for c in Agent[v]:
            lam_idx[(v, c, 'x')] = off
            lam_idx[(v, c, 'y')] = off + 1
            off += 2
    return agents, pos_idx, x_base, z_base, lam_base, lam_idx, off

def project_box(x, a, r):
    return np.array([np.clip(x[0], a[0]-r, a[0]+r),
                     np.clip(x[1], a[1]-r, a[1]+r)])

def dynamics(
    yy, yy_prev,
    Agent, Center, C, rho, A,
    tau, sigma, nu, alpha,
    r=None
):
    # ---- indices & unpack ----
    agents, pos_idx, x_base, z_base, lam_base, lam_idx, total_len = _make_indices(Agent)
    n = len(agents)
    x = np.array([yy[x_base + i] for i in range(2*n)])
    z = np.array([yy[z_base + i] for i in range(2*n)])
    x_prev = np.array([yy_prev[x_base + i] for i in range(2*n)])
    z_prev = np.array([yy_prev[z_base + i] for i in range(2*n)])

    lam = {v:{c: np.array([yy[lam_idx[(v,c,'x')]], yy[lam_idx[(v,c,'y')]]]) for c in Agent[v]} for v in agents}
    lam_prev = {v:{c: np.array([yy_prev[lam_idx[(v,c,'x')]], yy_prev[lam_idx[(v,c,'y')]]]) for c in Agent[v]} for v in agents}

    # ---- extrapolation ----
    x_til = x + alpha*(x - x_prev)
    z_til = z + alpha*(z - z_prev)
    lam_til = {v:{c: lam[v][c] + alpha*(lam[v][c]-lam_prev[v][c]) for c in Agent[v]} for v in agents}

    def get2(arr, v):
        ix, iy = pos_idx[v]
        return np.array([arr[ix], arr[iy]])
    def set2(arr, v, val):
        ix, iy = pos_idx[v]
        arr[ix], arr[iy] = val[0], val[1]

    # ---- update phase ----
    x_next = x_til.copy()
    z_next = z_til.copy()
    lam_next = {v:{} for v in agents}
    N = {v: _neighbors_for_agent(v, Agent, Center) for v in agents}

    # x-update (P_{Ω_i} = box projection if r given)
    for v in agents:
        xv = get2(x_til, v)
        neighbors = N[v]; deg = len(neighbors)
        sum_u = np.zeros(2)
        for u in neighbors:
            sum_u += get2(x_til, u)
        grad = 2*rho[v]*(xv - np.array(A[v]))
        if deg > 0:
            grad += 2*(deg*xv - sum_u)
        lam_sum = np.zeros(2)
        for c in Agent[v]:
            lam_sum += lam_til[v][c]          # A_i^T \tilde λ_{i,k} with A_i = I
        step = xv - tau[v]*(grad - lam_sum)    # x_{i,k+1}
        if (r is not None) and (v in r):
            step = project_box(step, np.array(A[v]), r[v])
        set2(x_next, v, step)

    # z-update: consensus on aggregated λ (w_ij = 1)
    lam_tot_til = {v: sum((lam_til[v][c] for c in Agent[v]), start=np.zeros(2)) for v in agents}
    for v in agents:
        zv = get2(z_til, v)
        cons = np.zeros(2)
        for j in N[v]:
            cons += (lam_tot_til[v] - lam_tot_til[j])
        set2(z_next, v, zv + nu[v]*cons)

    # λ-update (NO projection; A_i = I; b_i = -C[c] UNscaled)
    b_unscaled = {c: -np.array(C[c]) for c in Center}   # <-- per your requirement
    z_next_vec = {v: get2(z_next, v) for v in agents}
    z_til_vec  = {v: get2(z_til,  v) for v in agents}
    for v in agents:
        cons_z = np.zeros(2)
        cons_lam = np.zeros(2)
        for j in N[v]:
            cons_z   += 2*(z_next_vec[v] - z_next_vec[j]) - (z_til_vec[v] - z_til_vec[j])
            cons_lam += (lam_tot_til[v] - lam_tot_til[j])
        for c in Agent[v]:
            # with A_i = I and b_i = -C[c]
            data_term = (2*get2(x_next, v) - get2(x_til, v)) - b_unscaled[c]
            bracket = data_term + cons_z + cons_lam
            lam_next[v][c] = lam_til[v][c] - sigma[v]*bracket   # NO projection

    # ---- pack ----
    yy_next = np.zeros_like(yy)
    for v in agents:
        set2(yy_next, v, get2(x_next, v))
    for v in agents:
        ix, iy = pos_idx[v]
        yy_next[2*n + ix] = z_next[ix]
        yy_next[2*n + iy] = z_next[iy]
    for v in agents:
        for c in Agent[v]:
            yy_next[lam_idx[(v,c,'x')]] = lam_next[v][c][0]
            yy_next[lam_idx[(v,c,'y')]] = lam_next[v][c][1]
    return yy_next