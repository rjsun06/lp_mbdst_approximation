#%%
from typing import Literal
import numpy as np
import itertools
from lp_mbdst import linprog_MBDST
from weird_cases import weird_cases, bug_cases

#%% testing
def fully_connected(N):
    ne = N*(N-1)//2
    ind = np.array(list(itertools.combinations(range(N),2)))
    ret = np.zeros((N,ne))
    ret[ind[:,0],np.arange(ne)]=1
    ret[ind[:,1],np.arange(ne)]=1
    return ret


def gen_case(N=5,bound_c=None,bound_b=None,mode:Literal['fully','weird']='fully'):
    if bound_c is None: bound_c = (-N,N)
    if bound_b is None: bound_b = (1,N)
    if mode == 'fully':
        g = fully_connected(N)
        c = np.random.randint(*bound_c,g.shape[1])
        b = np.random.randint(*bound_b,g.shape[0])
    # elif mode == 'random':
    elif mode == 'weird':
        return weird_cases[N]
    elif mode == 'bug':
        return bug_cases[N]
    else:
        raise NotImplementedError('mode not available')
    return g,c,b

def is_feasible(g,c,b):
    x = linprog_MBDST(g,c,b) 
    return x is not None


#%% fetching
def get_edges(incidence_matrix):
    rows, cols = np.where(incidence_matrix.T == 1)
    edges = list(zip(cols[::2],cols[1::2]))
    return edges

def get_vertices(incidence_matrix):
    vertices = np.arange(incidence_matrix.shape[0])
    return(vertices)