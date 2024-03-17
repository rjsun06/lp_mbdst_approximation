#%%
import numpy as np
import itertools
from mbdst_kplus1 import MBDST1,MBDST2, MBDST3
from utils import *

#%%
def test_single_MBDST(fun,N):
    case = gen_case(N)
    # print(fun(*case))
    return None

def compare_results(funs,N):
    g,c,b = gen_case(N)
    ret = [f(g.copy(), c.copy(), b.copy()) for f in funs]
    for tmp in ret:
        if tmp is None and is_feasible(g,c,b):
            print('got None')
            print(ret)
            print(c,b)
            return
    ret = np.stack(ret)
    if np.logical_or.reduce(ret != ret[0],axis=0).any():
        print('got diff')
        print(ret)
        print(c,b)
    return None
#%%
def test():
    # test_single_MBDST(MBDST2,15)
    compare_results((MBDST1,MBDST2,MBDST3),5)
    # compare_results((MBDST2,),10)

from timeit import Timer
t = Timer("test()", "from __main__ import test")
print(t.timeit(10000))
# %%
