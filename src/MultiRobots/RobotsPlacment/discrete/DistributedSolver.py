import numpy as np

from Robotics.discrete.forward_backward_disc import dynamics

def solve(Agent, Center, C, rho, A, lr, r = None, TimeStamp = 50000, seed=0, x_init=None):
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
        k = 2
        length = k*len(Agent) + sum([2*len(Agent[v]) for v in Agent])
        x_init = np.squeeze(-3 + 1*np.random.rand(length, 1)*(1-(-3)))
    trajectory = [x_init]
    for _ in range(TimeStamp):
        x_init = dynamics(x_init, Agent, Center, C, rho, A, lr, r=r)
        trajectory.append(x_init)
    
    return TimeStamp, np.array(trajectory)