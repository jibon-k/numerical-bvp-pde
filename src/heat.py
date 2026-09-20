import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded


def ftcs(u0, r, nt):
    u = u0.copy()
    hist = [u.copy()]
    for k in range(nt):
        u[1:-1] = u[1:-1] + r*(u[:-2]-2*u[1:-1]+u[2:])
        hist.append(u.copy())
    return u, np.array(hist)


def crank_nicolson(u0, r, nt):
    n = len(u0)-2
    ab = np.zeros((2, n))
    ab[0, 1:] = -r/2
    ab[1, :] = 1+r
    c = cholesky_banded(ab)
    u = u0.copy()
    hist = [u.copy()]
    for k in range(nt):
        b = (1-r)*u[1:-1] + r/2*(u[:-2]+u[2:])
        u[1:-1] = cho_solve_banded((c, False), b)
        hist.append(u.copy())
    return u, np.array(hist)


def hat(x):
    return np.minimum(2*x, 2-2*x)
