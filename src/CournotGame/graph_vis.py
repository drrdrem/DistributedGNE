import matplotlib.pyplot as plt
import networkx as nx


def constructgraph(firm_G, market_G):
    '''
    Create a firm-market diagram for visualization.
    Args:
        firm_G (dict): Connections from agents to markets.
        market_G (dict): Connections from markets to agents.
    Returns:
        None
    '''
    B = nx.Graph()
    n = [v for v in firm_G]
    l = [str(m) for m in market_G]
    B.add_nodes_from(n, bipartite=0)
    B.add_nodes_from(l, bipartite=1)

    connection = []
    for v in firm_G:
        for j in firm_G[v]:
            connection.append((v, str(j)))

    B.add_edges_from(connection)
    top = nx.bipartite.sets(B)[0]
    pos = nx.bipartite_layout(B, top)
    node_color = ['green' for _ in firm_G] + ['blue' for _ in market_G]
    
    nx.draw_circular(B, node_color =node_color, with_labels = True)

    plt.axis('off')
