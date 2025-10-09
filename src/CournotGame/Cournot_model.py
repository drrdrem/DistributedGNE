import numpy as np

def generate_parameters(firm_G, market_G, seed=42):
    '''
    Generate Parameters for Cournot Competition with Arbitrary Graph.

    Args:
        firm_G (dict): Agent-to-market connections.
        market_G (dict): Market-to-agent connections.
        seed (int): Random seed for reproducibility, default is 42.
    returns:
        n (int): Total number of variables.
        P (float): Base price for the inverse demand law, randomly chosen between 250 and 500.
        d (float): Demand factor for the inverse demand law, randomly chosen between 1 and 5.
        c1v (float): Individual production cost coefficient, randomly selected between 1 and 4.
        c2v (float): Individual production cost coefficient, randomly selected between 1 and 8.
        r (float): Market energy requirement, randomly chosen between 20 and 80.
        theta (float): Upper production limit for each agent, default is None.
    '''
    np.random.seed(seed=seed)

    ni = [sum([1 for j in firm_G[i]]) for i in firm_G]
    
    m = len(market_G)
    N = len(firm_G)
    n = sum(ni)

    P = np.random.uniform(250, 500, m)
    d = np.random.uniform(1, 5, m)
    c1v = np.random.uniform(1, 4, n)
    c2v = np.random.uniform(1, 8, N)
    r = np.random.uniform(20, 80, m)
    theta = np.random.uniform(100, 200, n)

    return n, P, d, c1v, c2v, r, theta