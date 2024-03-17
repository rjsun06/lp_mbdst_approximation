#%%
from scipy.optimize import linprog
import numpy as np
from lp_mst import linprog_MST_constraints


def linprog_MBDST(graph, costs, degree_bounds, degree_bound_mask=None, edges_taken=None):
    """
    input:
        - graph: vertix-edge incidency matrix of shape |V|*|E|
        - cost: cost array of shape |E|
        - degree_bounds: degree bounds of shape |V|
    Bounded Degree Minimum Spanning Tree linear programming relaxation.

    Not guaranteed an integral solution!

    From LP-MST, add a constraint for each vertex that ensures its degree is within
    some specified bound (constrain sum of the edge indicators incident to it)
    """
    nv,ne = graph.shape
    degree_bounds = np.array(degree_bounds)
    if degree_bound_mask is None: degree_bound_mask = np.ones_like(degree_bounds).astype(bool)
    A_ub, b_ub, A_eq, b_eq = linprog_MST_constraints(graph)
    if edges_taken is not None:
        for i in np.where(edges_taken.flatten())[0]:
            A_eq = np.concatenate([A_eq,np.zeros((1,ne))],axis=0)
            A_eq[-1,i] = 1
            b_eq = np.append(b_eq,1)
    # Add degree bound constraints.
    A_ub = np.concatenate([A_ub,graph[degree_bound_mask]],axis=0)
    b_ub = np.concatenate([b_ub,degree_bounds[degree_bound_mask]],axis=0)
    bounds = (0, None)
    # print(A_eq,b_eq)
    # print(A_ub,b_ub)
    res = linprog(costs, A_ub, b_ub, A_eq, b_eq, bounds, method="highs-ds")

    sol = res.x
    fval = res.fun
    return sol, fval

if __name__ == "__main__":
    g =[[1, 1, 0, ], \
        [1, 0, 1, ], \
        [0, 1, 1, ]]
    c = [2, 4, 4, ] 
    b = [1, 4, 2]
    w = [1, 1, 1]
    f = [0, 1, 1]
    g,c,b,w,f = list(map(np.array,[g,c,b,w,f]))
    
    sol, fval = linprog_MBDST(g, c, b, w, f)
    print(fval, sol)
