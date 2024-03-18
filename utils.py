#%%
from typing import Literal
import numpy as np
import itertools
from lp_mbdst import linprog_MBDST

#%% testing
def fully_connected(N):
    ne = N*(N-1)//2
    ind = np.array(list(itertools.combinations(range(N),2)))
    ret = np.zeros((N,ne))
    ret[ind[:,0],np.arange(ne)]=1
    ret[ind[:,1],np.arange(ne)]=1
    return ret

def gen_case(N,mode:Literal['fully']='fully'):
    if mode == 'fully':
        g = fully_connected(N)
        c = np.random.randint(-N,N,N*(N-1)//2)
        b = np.random.randint(1,N,N)
    # elif mode == 'random':
    else:
        raise NotImplementedError("Use mode = 'fully' for now.")
    return g,c,b

def is_feasible(g,c,b):
    x, _ = linprog_MBDST(g,c,b) 
    return x is not None


#%% fetching
def get_edges(incidence_matrix):
    rows, cols = np.where(incidence_matrix.T == 1)
    edges = list(zip(cols[::2],cols[1::2]))
    return edges

def get_vertices(incidence_matrix):
    vertices = np.arange(incidence_matrix.shape[0])
    return(vertices)