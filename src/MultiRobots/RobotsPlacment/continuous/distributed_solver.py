from scipy.integrate import solve_ivp
import numpy as np

from Robotics.primal_dual import dynamics

def solve(Agent, Center, C, rho, A, r = None, Tmax = 50, TimeStamp = 50000, seed=0, x_init=None):
    """
    Distributed Solver for Algorithm 1.

    Args:
        Agent (float): .
        Center (float): .
        C (float): .
        rho (float): .
        A (float): .
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
        print("Here")
        k = 2 if not r else 6
        length = k*len(Agent) + sum([2*len(Agent[v]) for v in Agent])
        x_init = np.squeeze(-3 + 1*np.random.rand(length, 1)*(1-(-3)))

    t_span = np.linspace(0, Tmax, TimeStamp)
    sol = solve_ivp(dynamics, [0, Tmax], 
                np.squeeze(x_init),
                args=(Agent, Center, C, rho, A, r), dense_output=True)
    
    return t_span, sol.sol(t_span).T