import numpy as np
from scipy.linalg import solve_banded


def rayleigh_ritz(ne, f, q):
    h = 1/ne
    x = np.linspace(0, 1, ne+1)
    n = ne-1
    ke = (1/h)*np.array([[1, -1], [-1, 1]]) + q*(h/6)*np.array([[2, 1], [1, 2]])
    # 3-point gauss on [-1,1]
    gp = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    gw = np.array([5/9, 8/9, 5/9])
    dg = np.zeros(n)
    up = np.zeros(n-1)
    b = np.zeros(n)
    for e in range(ne):
        xm = x[e] + h/2
        xq = xm + gp*h/2
        fq = f(xq)
        phi1 = (x[e+1]-xq)/h
        phi2 = (xq-x[e])/h
        b1 = np.sum(gw*fq*phi1)*h/2
        b2 = np.sum(gw*fq*phi2)*h/2
        # node e is index e-1 (interior)
        i, j = e-1, e
        if i >= 0:
            dg[i] += ke[0, 0]
            b[i] += b1
        if j < n:
            dg[j] += ke[1, 1]
            b[j] += b2
        if i >= 0 and j < n:
            up[i] += ke[0, 1]
    ab = np.zeros((3, n))
    ab[0, 1:] = up
    ab[1, :] = dg
    ab[2, :-1] = up
    c = solve_banded((1, 1), ab, b)
    y = np.zeros(ne+1)
    y[1:-1] = c
    return x, y
