import numpy as np
from numpy import array
weird_cases = [
#only IP gives different sol
(np.array([[1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 1., 1., 1., 0., 0., 0.],
        [0., 1., 0., 0., 1., 0., 0., 1., 1., 0.],
        [0., 0., 1., 0., 0., 1., 0., 1., 0., 1.],
        [0., 0., 0., 1., 0., 0., 1., 0., 1., 1.]]), 
        np.array([-1, -1, -2,  4, -3,  4,  3, -5,  2,  2]), 
        np.array([1, 4, 4, 1, 1])), 
]
bug_cases = [
# MBDST2 gives larger tree then IP
(array([[1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1., 0.],
       [0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 1.],
       [0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1., 1.]]), array([-2,  5,  2,  5, -2,  5, -3, -1, -4,  0, -5, -2,  4,  3,  1]), array([1, 2, 4, 2, 3, 2]))
#only MBDST1 gives None
(np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1.,
        1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
        0., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
        0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1.,
        1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
        0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
        0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
        0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
        0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
        1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.,
        0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
        1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
        0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1., 1.]]), 
        np.array([ -5,  -2,   1,   8,   5,   7,   5,   0, -10,  -8,  -3,   1,   2,
        -8,   5,  -7,  -4,   2,   8,   6,   8,   9,  -7,   0,   1,   6,
         9,  -1,   2,  -2,  -4,  -6,  -8,   8,  -9,  -3,   2,  -5,   0,
        -2,   3,  -6,   5,  -1,  -4]), 
        np.array([1, 2, 1, 4, 3, 3, 7, 6, 2, 3])),
(array([[1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1., 0.],
       [0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 1.],
       [0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1., 1.]]), array([-4, -2, -6,  0,  1,  4,  2,  1, -2, -3, -1, -3, -6,  2,  2]), array([1, 4, 1, 1, 3, 2]))
]