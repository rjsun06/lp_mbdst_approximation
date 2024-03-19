#%%
import numpy as np
from mbdst import MBDST_true, MBDST_LP, MBDST1, MBDST2, MBDST3
from utils import *
from timeit import Timer

_log = print
def set_log(fun):
    global _log
    _log = fun
#%%
def test_single_MBDST(fun,N):
    case = gen_case(N)
    ret = fun(*case)
    return ret

def compare_results(funs,case=gen_case(5),report_degrees=False,report_costs=False):
    g,c,b = case
    ret = [f(g.copy(), c.copy(), b.copy()) for f in funs]
    for _ in (x for x in ret if x is None):
        return -1,{
            'names':[f.__name__ for f in funs],
            'x':ret,
            'case':(g,c,b)}
    if np.logical_or.reduce(np.stack(ret) != ret[0],axis=0).any():
        return 1,{
                'names':[f.__name__ for f in funs],
                'x':ret,
                'case':(g,c,b)}
    return 0,{
                'names':[f.__name__ for f in funs],
                'x':ret,
                'case':(g,c,b)}

def report(stat,dic):
    g,c,b = dic['case']
    x = np.stack([x if x is not None else np.full_like(c,fill_value=0) for x in dic['x'] ])
    if stat == 0:
        return
    if stat == 1:
        _log('got diff')
    if stat == -1:
        if is_feasible(*dic['case']):
            _log('got None BUT FEASIBLE')
        else:
            _log('got None as expected')
            _log('bounds:',b)
            return
    _log('x:')
    _log(x)
    _log('costs:')
    _log(c@x.T)
    _log('d bounds:')
    _log(b.astype(float))
    _log('degres:')
    _log(x@g.T)
    _log(dic['case'])
    return
#%%
def test():
    # test_single_MBDST(MBDST2,15)
    # compare_results((MBDST_true,MBDST1,MBDST2,MBDST3),5,report_degrees=True)
    # compare_results((MBDST_LP,MBDST_true,MBDST1,MBDST2,MBDST3),7)
    report(*compare_results((MBDST_LP,MBDST_true,MBDST1,MBDST2,MBDST3),
                            gen_case(10)))
    # compare_results((MBDST2,),10)


t = Timer("test()", "from __main__ import test")
print(t.timeit(100))
# %%
report(*compare_results((MBDST_LP,MBDST_true,MBDST1,MBDST2,MBDST3),
                        gen_case(mode='bug',N=0)))
