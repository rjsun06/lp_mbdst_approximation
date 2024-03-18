#%%
import matplotlib.pyplot as plt
import networkx as nx
from utils import *


#%%
def make_graph(incidence_matrix,edge_costs):
    # Create a graph object
    G = nx.Graph()

    # Add nodes to the graph
    G.add_nodes_from(get_vertices(incidence_matrix))

    # Add edges to the graph based on the incidence matrix
    edges = get_edges(incidence_matrix)

    G.add_edges_from(zip(*zip(*edges),map(lambda a:{'cost':a},edge_costs)))
    return G

def fix_pos(G):
    return nx.spring_layout(G)

def visualize_graph(G,pos=None,stengthen_edges=None, **kwe):

    # Create a dictionary for edge labels
    if pos is None: pos = nx.spring_layout(G)
    costs = nx.get_edge_attributes(G, 'cost')
    nx.draw(G,pos,with_labels=True,**kwe)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[e for i,e in enumerate(G.edges) if stengthen_edges[i]],
        width=8,
        alpha=0.5,
        edge_color="tab:red",
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=costs)

incidence_matrix = np.array([
    [1, 1, 0, 0, 0],
    [1, 0, 1, 1, 0],
    [0, 1, 1, 0, 1],
    [0, 0, 0, 1, 1]
])
edge_costs = [2, 3, 1, 4, 5]
select_edges = np.array([1, 0, 1, 1, 1]).astype(bool)
G = make_graph(incidence_matrix,edge_costs)
pos = fix_pos(G)
visualize_graph(G,pos,stengthen_edges=select_edges)
plt.show()
select_edges = np.array([1, 0, 0, 0, 1]).astype(bool)
visualize_graph(G,pos,stengthen_edges=select_edges)
plt.show()

# %%
