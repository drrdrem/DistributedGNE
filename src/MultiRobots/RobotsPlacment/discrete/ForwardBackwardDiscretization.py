# Dynamics with L-infinity (box) constraints and plotting of per-iteration convergence.
# This cell defines:
# - dynamics(): one-step update using your objective and extragradient dual update
import numpy as np

def project_box(x, a, r):
    return np.array([np.clip(x[0], a[0]-r, a[0]+r),
                     np.clip(x[1], a[1]-r, a[1]+r)])

def dynamics(yy, Agent, Center, C, rho, A, lr, r=None):
    """
    One dynamics step with L-infinity (box) constraints:
      min_v  rho_v ||z_v - A_v||^2 + sum_{u in N(v)} ||z_v - z_u||^2
      s.t.   |z_v[i] - A_v[i]| <= r_v   for i in {x,y}  (box)
             shared constraints g_c(z) = sum_{a in Center[c]} z_a + b = 0, b=-|Center[c]|*C[c]
    Extragradient dual: lambda^{k+1} = lambda^k + sigma * (2 g(z^{k+1}) - g(z^k))
    Args follow your earlier convention (dicts for Agent, Center, etc.).
    """
    agents = list(Agent.keys())
    n = len(agents)

    # index maps for positions and lambdas
    pos_idx = {v: (2*i, 2*i+1) for i, v in enumerate(agents)}
    lam_base = 2*n
    lam_idx = {}
    off = lam_base
    for v in agents:
        for c in Agent[v]:
            lam_idx[(v, c, 'x')] = off
            lam_idx[(v, c, 'y')] = off + 1
            off += 2

    xs = [0.0]*(2*n)
    las = []

    # ---- primal update ----
    for v in agents:
        ix, iy = pos_idx[v]
        zvx, zvy = yy[ix], yy[iy]

        # neighbors (dedup) from constraints involving v
        neighbors = set().union(*(Center[c] for c in Agent[v])) if Agent[v] else set()
        neighbors.discard(v)
        deg = len(neighbors)

        # gradient of rho_v ||z_v - A_v||^2 + sum_u ||z_v - z_u||^2
        dx = 2*rho[v]*(zvx - A[v][0])
        dy = 2*rho[v]*(zvy - A[v][1])
        if deg > 0:
            sum_ux = sum(yy[pos_idx[u][0]] for u in neighbors)
            sum_uy = sum(yy[pos_idx[u][1]] for u in neighbors)
            dx += 2*(deg*zvx - sum_ux)
            dy += 2*(deg*zvy - sum_uy)

        # add dual terms once per constraint involving v
        for c in Agent[v]:
            dx += yy[lam_idx[(v, c, 'x')]]
            dy += yy[lam_idx[(v, c, 'y')]]

        # gradient step
        zpx = zvx - lr[v][0]*dx
        zpy = zvy - lr[v][0]*dy

        # L-infinity (box) projection around anchor A[v] with half-width r[v]
        if (r is not None) and (v in r):
            ax, ay = A[v]
            rv = r[v]
            zpx = project_box(zpx, ax, rv)
            zpy = project_box(zpy, ay, rv)

        xs[ix], xs[iy] = zpx, zpy

    # ---- dual update (extragradient) ----
    for v in agents:
        for c in Agent[v]:
            m = len(Center[c])
            b0 = -m * C[c][0]
            b1 = -m * C[c][1]

            gx  = sum(yy[pos_idx[a][0]] for a in Center[c])
            gy  = sum(yy[pos_idx[a][1]] for a in Center[c])
            gnx = sum(xs[pos_idx[a][0]] for a in Center[c])
            gny = sum(xs[pos_idx[a][1]] for a in Center[c])

            egx = 2*(gnx + b0) - (gx + b0)   # = 2*gnx - gx + b0
            egy = 2*(gny + b1) - (gy + b1)   # = 2*gny - gy + b1

            lamx = yy[lam_idx[(v, c, 'x')]] + lr[v][1]*egx
            lamy = yy[lam_idx[(v, c, 'y')]] + lr[v][1]*egy
            las.extend([lamx, lamy])

    return np.array(xs + las)