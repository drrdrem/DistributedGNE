from scipy.integrate import solve_ivp
import numpy as np


def dynamics(t, yy, firm_G, market_G, c2v, c1v, P, d, r, theta):
    '''Algorithm 1: Distributed Primal-Dual Dynamics.
    Args:
        t (list): Time range [t_init, t_final].
        yy (list): Vector field from the previous step.
        firm_G (dict): Agent-to-market connections.
        market_G (dict): Market-to-agent connections.
        P (float): Base price for the inverse demand law, randomly chosen between 250 and 500.
        d (float): Demand factor for the inverse demand law, randomly chosen between 1 and 5.
        c1v (float): Individual production cost coefficient, randomly selected between 1 and 4.
        c2v (float): Individual production cost coefficient, randomly selected between 1 and 8.
        r (float): Market energy requirement, randomly chosen between 20 and 80.
        theta (float): Upper production limit for each agent, default is None.
    Returns:
        f (list): Updated vector field.
    '''
    N = len(firm_G)

    n_v = [len(firm_G[v]) for v in firm_G]
    n = sum(n_v)

    n_v_loc = [sum(n_v[0:i]) for i in range(len(n_v))]
    paras = {v:[n_v_loc[v]+j for j in range(n_v[v])] + [n+n_v_loc[v]+j for j in range(n_v[v])] + [2*n+n_v_loc[v]+j for j in range(n_v[v])]+ [3*n +n_v_loc[v]+j for j in range(n_v[v])] for v in firm_G}
    f = [0 for _ in range(paras[N-1][-1]+1)]
    for i in paras:
        g_c = 2*c2v[i]*sum([yy[j] for j in paras[i][0:n_v[i]]]) # Production Cost
        for j in range(n_v[i]):
            price_idx = firm_G[i][j]

            # The dynamics of z
            x_idx = paras[i][j]
            L_gx = g_c + c1v[paras[i][j]]- (P[price_idx]-d[price_idx]*sum([yy[paras[fidx][firm_G[fidx].index(price_idx)]] for fidx in market_G[price_idx]]+[yy[paras[i][j]]]))
            L_gx += yy[paras[i][n_v[i]+j]] - yy[paras[i][2*(n_v[i])+j]] + yy[paras[i][3*(n_v[i])]]
            f[x_idx] = -L_gx

            # The dynamics of lamda
            la_idx = paras[i][n_v[i]+j]
            L_gla = sum([yy[paras[fidx][firm_G[fidx].index(price_idx)]] for fidx in market_G[price_idx]])-r[price_idx]

            f[la_idx] = L_gla

            # Non-Negative
            mu_idx = paras[i][2*(n_v[i])+j]
            L_gmu = ((-yy[x_idx]>0)|(yy[mu_idx]>0))*(-yy[x_idx])
            f[mu_idx] = L_gmu

            # Upper limit
            mu_idx = paras[i][3*(n_v[i])+j]
            L_gmu = ((yy[x_idx]-theta[x_idx]>0)|(yy[mu_idx]>0))*(yy[x_idx]-theta[x_idx])
            f[mu_idx] = L_gmu

    return f