import numpy as np
import itertools

from lp_mbdst import linprog_MBDST

def fully_connected(N):
    ne = N*(N-1)//2
    ind = np.array(list(itertools.combinations(range(N),2)))
    ret = np.zeros((N,ne))
    ret[ind[:,0],np.arange(ne)]=1
    ret[ind[:,1],np.arange(ne)]=1
    return ret

def gen_case(N):
    g = fully_connected(N)
    c = np.random.randint(-N,N,N*(N-1)//2)
    b = np.random.randint(1,N,N)
    return g,c,b

def is_feasible(g,c,b):
    x, _ = linprog_MBDST(g,c,b) 
    return x is not None