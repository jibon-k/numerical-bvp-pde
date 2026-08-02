import numpy as np
from scipy.linalg import solve_banded


def linear_bvp(f, a, b, ya, yb, N):
    h = (b-a)/N
    x = np.linspace(a, b, N+1)
    n = N-1
    ab = np.zeros((3, n))
    ab[0, 1:] = 1
    ab[1, :] = -2
    ab[2, :-1] = 1
    rhs = h**2*f(x[1:-1])
    # bounday values
    rhs[0] -= ya
    rhs[-1] -= yb
    y = np.zeros(N+1)
    y[0], y[-1] = ya, yb
    y[1:-1] = solve_banded((1, 1), ab, rhs)
    return x, y


def newton_bvp(N, ya, yb, y0=None, tol=1e-9, maxit=30):
    h = 1/N
    x = np.linspace(0, 1, N+1)
    if y0 is None:
        y = ya + (yb-ya)*x
    else:
        y = y0(x).astype(float)
    y[0], y[-1] = ya, yb
    n = N-1
    res = []
    for k in range(maxit):
        F = (y[:-2]-2*y[1:-1]+y[2:])/h**2 - 1.5*y[1:-1]**2
        res.append(np.linalg.norm(F, np.inf))
        if res[-1] < tol:
            break
        ab = np.zeros((3, n))
        ab[0, 1:] = 1/h**2
        ab[1, :] = -2/h**2 - 3*y[1:-1]
        ab[2, :-1] = 1/h**2
        d = solve_banded((1, 1), ab, -F)
        y[1:-1] += d
    return x, y, res
