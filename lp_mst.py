#%%
from scipy.optimize import linprog

from itertools import product

import numpy as np

def linprog_MST_constraints(graph):
    """
    input:
        - graph: vertix-edge incidency matrix of shape |V|*|E|
    return: 
        - A_ub: array of (1,|E|)
        - b_ub: array of (|V|-1)
        - A_eq: array of (N,|E|) for some N<=2**|V|-|V|
        - b_eq: array of (N)
    explain:
        generating the constrains following scipy.optimize.linprog 
        format standing for:
        - sum(E(V))  = |V| - 1
                    ; Spanning tree has |V|-1 edges

        - sum(E(S)) <= |S| - 1    forall subsets S of V
                    ; No subset can be denser than a tree.
    """
    nv, ne = graph.shape

    A_eq = np.logical_or.reduce(graph,axis=0,keepdims=True).astype(int)
    b_eq = [graph.shape[0]-1]

    powerset = np.array(list(product([0, 1], repeat=nv)))
    A_ub = (powerset @ graph == 2).astype(int)
    b_ub = np.sum(powerset, axis=1)-1
    mask = np.sum(A_ub,axis=1)>0
    A_ub = A_ub[mask]
    b_ub = b_ub[mask]
    # print(f"{len(A_ub)} subset constraints.")
    return A_ub, b_ub, A_eq, b_eq

def linprog_MST(graph,costs):
    bounds = (0, None)
    res = linprog(costs, *linprog_MST_constraints(graph), bounds, method="highs-ds")
    sol = res.x
    fval = res.fun
    return sol, fval

if __name__ == "__main__":
    vertices = list(range(4))
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (2, 3),
    ]
    costs = [
        1,
        1.1,
        6,
        3,
        5,
    ]
    # costs = [1]*5
    edges=np.array(edges)
    graph = np.zeros((len(vertices),len(edges)))
    graph[edges[:,0],np.arange(len(edges))]=1
    graph[edges[:,1],np.arange(len(edges))]=1
    sol, fval = linprog_MST(graph,costs)
    print(fval, sol)
    for i, v in enumerate(sol):
        v = round(v, 4)
        if v == 1:
            print(edges[i])
