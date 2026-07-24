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
