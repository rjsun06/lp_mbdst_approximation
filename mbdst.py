#%%

import numpy as np
from lp_mbdst import linprog_MBDST
from utils import *
# LP = linprog_MBDST

def MBDST_true(G,C,B):
    return linprog_MBDST(G,C,B,integral=True)

def MBDST_LP(G,C,B):
    return linprog_MBDST(G,C,B)

def MBDST1_step(G, C, B):
    LP = linprog_MBDST
    nv, ne = G.shape
    #1
    T = np.zeros(ne).astype(bool)
    W = np.ones(nv).astype(bool)
    V = np.ones(nv).astype(bool)
    #2
    def fun():
        nonlocal G,C,B,T,W
        while G.any(): 
            #2a
            e = LP(G[V],C,B[V],W[V])
            if e is None: yield None
            # print(e)
            G[:,e==0]=0
            C[e==0]=0
            #2b
            v = (np.sum(G,axis=1) <= 1)
            s = np.logical_or.reduce(G[v,:],axis=0)
            T[s] = 1
            # G, W, B = [np.delete(arr, v, axis=0) for arr in (G, W, B)]
            V[v]=0
            B = B - (G*V[:,None])@s
            # G[v]=0
            # W[v]=0
            # B[v]=0
            G[:,s]=0
            C[s]=0
            #c
            v = np.sum(G,axis=1) <= 3
            W[v] = 0
            yield np.sum(G,axis=0).astype(bool),e,W
    #3
    return T,fun

def MBDST2_step(G:np.ndarray, C, B):
    LP = linprog_MBDST
    nv, ne = G.shape
    #1
    W = np.ones(nv).astype(bool)
    e = np.zeros(ne)
    #2
    def fun():
        nonlocal G,C,B,e,W
        while W.sum()>0: 
            #2a
            e[:]= LP(G,C,B,W)
            if e is None: yield None
            G[:,e==0] = 0
            C[e==0] = 0
            #2b
            v = np.sum(G,axis=1) <= B+1 
            if (W[v] == 0).all(): break
            W[v] = 0
            yield np.sum(G,axis=0).astype(bool),e,W
        #3
        e[:] = LP(G,C,B)
        yield np.sum(G,axis=0).astype(bool),e,W
    #4
    return e,fun

def MBDST3_step(G:np.ndarray, C, B):
    LP = linprog_MBDST
    nv, ne = G.shape
    #1
    F = np.zeros(ne).astype(bool)
    W = np.ones(nv).astype(bool)
    #2
    def fun():
        nonlocal G,C,B,F,W
        while F.sum()<nv-1: 
            #2a
            e = LP(G,C,B,W,F)
            if e is None: yield None
            G[:,e==0] = 0
            C[e==0] = 0
            #2b
            F[e==1] = 1
            #2c
            v = (G @ (np.logical_and(0<e, e<1))) <=2
            # if (W[v] == 0).all(): break
            W[v] = 0
            yield np.sum(G,axis=0).astype(bool),e,W
    #3
    return F,fun

def final(fun):
    def ret(*args):
        T, run = fun(*args)
        for ret in run(): 
            if ret is None: return None 
        return T.astype(int)
    return ret
MBDST1 = final(MBDST1_step)
MBDST2 = final(MBDST2_step)
MBDST3 = final(MBDST3_step)

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
        feasible = is_feasible(g.copy(),c.copy(),b.copy())
        for alg in [MBDST1, MBDST2, MBDST3]:
            sol = alg(g.copy(), c.copy(), b.copy())
            if sol is None: 
                if feasible:
                    print(None, 'BUT FEASIBLE.')
                else:
                    print(None, 'as desired')
                continue
            fval = c @ sol
            deg = g @ sol
            print(fval, sol, deg)
# %%
