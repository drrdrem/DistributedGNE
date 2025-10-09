from scipy.integrate import solve_ivp
import numpy as np

from CournotGame.primal_dual import dynamics

def solve(n, P, d, c1v, c2v, r, theta, firm_G, market_G, Tmax = 100, TimeStamp = 100, seed=0, x_init=None):
    """
    Distributed Solver for Algorithm 1.

    Args:
        t (list): Time range [t_init, t_final].
        yy (list): Vector field from the previous step.
        n (int): Total number of variables.
        P (float): Base price for the inverse demand law, randomly chosen between 250 and 500.
        d (float): Demand factor for the inverse demand law, randomly chosen between 1 and 5.
        c1v (float): Individual production cost coefficient, randomly selected between 1 and 4.
        c2v (float): Individual production cost coefficient, randomly selected between 1 and 8.
        r (float): Market energy requirement, randomly chosen between 20 and 80.
        theta (float): Upper production limit for each agent, default is None.
        Tmax (int): Maximum simulation time, default is 100.
        TimeStamp (int): Number of time steps in the simulation, default is 100.
        seed (int): Random seed for reproducibility, default is 0.
        x_init (np.array): Initial values, default is None.

    Returns:
        t_span (np.array): Time grid.
        sol (np.array): Numerical solution for the trajectory.
    """
    np.random.seed(seed=seed)
    if type(x_init)==type(None):
        x_init = 100*np.random.rand(4*(n), 1)

    t_span = np.linspace(0, Tmax, TimeStamp)
    sol = solve_ivp(dynamics, [0, Tmax], 
                np.squeeze(x_init),
                args=(firm_G, market_G, c2v, c1v, P, d, r, theta),
                rtol=1e-12, dense_output=True)
    
    return t_span, sol.sol(t_span).T