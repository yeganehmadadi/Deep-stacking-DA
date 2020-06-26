import numpy as np
from scipy.linalg import orth
#from exact_alm_lrr_l21v2 import exact_alm_lrr_l21v2
#from exact_alm_lrr_l1v2 import exact_alm_lrr_l1v2
#from inexact_alm_lrr_l21 import inexact_alm_lrr_l21
from inexact_alm_lrr_l1 import inexact_alm_lrr_l1


def solve_lrr(X, A, lamb, alm_type, display=False):
    Q = orth(A.T)
    B = A.dot(Q)


    Z, E = inexact_alm_lrr_l1(X, B, lamb, display)

    Z = Q.dot(Z)
    return (Z, E)

