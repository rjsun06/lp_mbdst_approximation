#%%

import numpy as np
from lp_mbdst import linprog_MBDST
from utils import *


def MBDST1(G, C, B):
    LP = linprog_MBDST
    nv, ne = G.shape
    #1
    T = np.zeros(ne).astype(bool)
    W = np.ones(nv).astype(bool)
    #2
    while np.sum(G)>0: 
        #2a
        e,_ = LP(G,C,B,W)
        if e is None: return None
        # print(e)
        G[:,e==0]=0
        C[e==0]=0
        #2b
        v = np.sum(G,axis=1) <= 1 
        e = np.logical_or.reduce(G[v,:],axis=0)
        T[e] = 1
        G = G[np.logical_not(v),:]
        W = W[np.logical_not(v)]
        B = B[np.logical_not(v)]
        B = B - np.sum(G,axis=1)
        G[:,e]=0
        C[e]=0
        #c
        v = np.sum(G,axis=1) <= 3
        W[v] = 0
    #3
    return T.astype(int)

def MBDST2(G:np.ndarray, C, B):
    LP = linprog_MBDST
    nv, ne = G.shape
    #1
    W = np.ones(nv).astype(bool)
    #2
    while W.sum()>0: 
        #2a
        e,_ = LP(G,C,B,W)
        if e is None: return None
        G[:,e==0] = 0
        C[e==0] = 0
        #2b
        v = np.sum(G,axis=1) <= B+1 
        if (W[v] == 0).all(): break
        W[v] = 0
    #3
    # print(G,C)
    e,_ = LP(G,C,B,W)
    #4
    return e.astype(int)

def MBDST3(G:np.ndarray, C, B):
    LP = linprog_MBDST
    nv, ne = G.shape
    #1
    F = np.zeros(ne).astype(bool)
    W = np.ones(nv).astype(bool)
    #2
    while F.sum()<nv-1: 
        #2a
        e,_ = LP(G,C,B,W,F)
        if e is None: return None
        G[:,e==0] = 0
        C[e==0] = 0
        #2b
        F[e==1] = 1
        #2c
        v = (G @ (np.logical_and(0<e, e<1))) <=2
        # if (W[v] == 0).all(): break
        W[v] = 0
    #3
    return F.astype(int)


if __name__ == "__main__":
    cases = [
        [
            fully_connected(4),
            [0, 0, -2, 0, -4, -3,],
            [1, 3, 2 ,2,],
        ],
        [
            fully_connected(6),
            [4, 1, 3, -5, 2, 4, 5, 1, -2, -4, 5, -3, 0, 3, -3],
            [3, 2, 4, 1, 2, 2],
        ],
        [
            fully_connected(6),
            [5, 3, -4, -3, 3, 3, -1, 4, 3, 2, -6, -1, 2, 4, -4],
            [1, 1, 2, 1, 1, 2]
        ],
    ]
    for g,c,b in cases:
        g,c,b = np.array(g), np.array(c), np.array(b)
        for alg in [MBDST1, MBDST2, MBDST3]:
            sol = alg(g.copy(), c.copy(), b.copy())
            if sol is None: 
                print(None)
                continue
            fval = c @ sol
            deg = g @ sol
            print(fval, sol, deg)