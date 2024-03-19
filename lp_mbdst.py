#%%
from typing import Literal
from scipy.optimize import linprog
import numpy as np
from lp_mst import linprog_MST_constraints
import pulp

pulp.LpSolverDefault.msg = False

def scipy_solver(*args,integral):
    if integral:
        raise Exception('scipy do not support IP')
    return linprog(*args, method="highs-ds")

def pulp_solver(costs, A_ub, b_ub, A_eq, b_eq, bounds, integral=False):
    problem = pulp.LpProblem("MinimumBoundedDegreeSpanningTree", pulp.LpMinimize)

    cat = 'Integer' if integral else pulp.LpContinuous
    x = pulp.LpVariable.dicts("x", list(range(len(costs))), 
                              lowBound=bounds[0], upBound=bounds[1], 
                              cat=cat)

    x=np.array(list(x.values()))
    # x=list(x.values())

    problem += pulp.lpSum(costs @ x)

    for i in range(b_ub.shape[0]):
        problem += pulp.lpSum(A_ub[i] @ x) <= b_ub[i]
    for i in range(b_eq.shape[0]):
        problem += pulp.lpSum(A_eq[i] @ x) == b_eq[i]

    # problem += np.sum(pulp.lpSum(A_ub @ x) <= b_ub)
    # problem += np.sum(pulp.lpSum(A_eq @ x) == b_eq)

    # solver = pulp.getSolver('CPLEX_CMD')
    # problem.solve(solver)
    problem.solve()
 
    if problem.status != pulp.LpStatusOptimal: return None

    return np.array([var.varValue if var.varValue else 0 for var in x])

solver = pulp_solver
# solver = scipy_solver
def linprog_MBDST(graph, costs, degree_bounds, degree_bound_mask=None, edges_taken=None, integral=False):
    #sanity
    graph = np.asarray(graph)
    costs = np.asarray(costs)
    nv,ne = graph.shape

    costs = np.asarray(costs)
    degree_bounds = np.asarray(degree_bounds)

    if degree_bound_mask is None: degree_bound_mask = np.ones(nv)
    degree_bound_mask = np.asarray(degree_bound_mask).astype(bool)

    if edges_taken is None: edges_taken = np.zeros(ne)
    edges_taken = np.asarray(edges_taken).astype(bool)
    constraints = constraints_MBDST(graph, costs, degree_bounds, degree_bound_mask, edges_taken)
    res = solver(*constraints,integral = integral)

    return res

def constraints_MBDST(graph, costs, degree_bounds, degree_bound_mask=None, edges_taken=None):
    nv,ne = graph.shape
    #main 
    A_ub, b_ub, A_eq, b_eq = linprog_MST_constraints(graph)

        # Add taken edge constraints.
    A_eq = np.concatenate([A_eq,np.identity(ne)[edges_taken]],axis=0)
    b_eq = np.concatenate([b_eq,np.ones(edges_taken.sum())],axis=0)

        # Add degree bound constraints.
    A_ub = np.concatenate([A_ub,graph[degree_bound_mask]],axis=0)
    b_ub = np.concatenate([b_ub,degree_bounds[degree_bound_mask]],axis=0)

        #solve
    bounds = (0, None)

    res = costs, A_ub, b_ub, A_eq, b_eq, bounds
        #report
    return res

if __name__ == "__main__":
    g =[[1, 1, 0, ], \
        [1, 0, 1, ], \
        [0, 1, 1, ]]
    c = [2, 4, 4, ] 
    b = [1, 4, 2]
    w = [1, 1, 1]
    f = [0, 1, 1]
    # g,c,b,w,f = list(map(np.array,[g,c,b,w,f]))
    
    sol = linprog_MBDST(g, c, b, w, f)
    if sol is None:
        print(sol)
    else:
        fval = c @ sol
        print(fval, sol)

    g = [[1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 1., 1., 1., 0., 0., 0.],
        [0., 1., 0., 0., 1., 0., 0., 1., 1., 0.],
        [0., 0., 1., 0., 0., 1., 0., 1., 0., 1.],
        [0., 0., 0., 1., 0., 0., 1., 0., 1., 1.]]
    c = [-1, -2, -5, -4,  3,  4,  0,  1, -1,  0]
    b = [2, 2, 4, 3, 2]
    sol = linprog_MBDST(g, c, b)
    soli = linprog_MBDST(g, c, b, integral= True)
    if sol is None or soli is None:
        print(sol,soli)
    else:
        fval = c @ sol
        fvali = c @ soli
        print(sol, fval)
        print(soli, fvali)
# %%
