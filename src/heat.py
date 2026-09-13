import numpy as np


def ftcs(u0, r, nt):
    u = u0.copy()
    hist = [u.copy()]
    for k in range(nt):
        u[1:-1] = u[1:-1] + r*(u[:-2]-2*u[1:-1]+u[2:])
        hist.append(u.copy())
    return u, np.array(hist)


def hat(x):
    return np.minimum(2*x, 2-2*x)
