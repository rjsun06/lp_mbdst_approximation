#%%
import numpy as np
import itertools
from mbdst_kplus1 import MBDST1,MBDST2, MBDST3
from utils import *

_log = print
def set_log(fun):
    global _log
    _log = fun
#%%
def test_single_MBDST(fun,N):
    case = gen_case(N)
    ret = fun(*case)
    return ret

def compare_results(funs,N):
    g,c,b = gen_case(N)
    ret = [f(g.copy(), c.copy(), b.copy()) for f in funs]
    for tmp in ret:
        if tmp is None and is_feasible(g,c,b):
            _log('got None')
            _log(ret)
            _log(c,b)
            return ret,-1
    ret = np.stack(ret)
    if np.logical_or.reduce(ret != ret[0],axis=0).any():
        _log('got diff')
        _log(ret)
        _log(c,b)
        return ret,1
    return ret,0
#%%
def test():
    # test_single_MBDST(MBDST2,15)
    compare_results((MBDST1,MBDST2,MBDST3),5)
    # compare_results((MBDST2,),10)

from timeit import Timer
t = Timer("test()", "from __main__ import test")
print(t.timeit(10000))
# %%
