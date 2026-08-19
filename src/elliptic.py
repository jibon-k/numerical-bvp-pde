import numpy as np
from scipy.sparse import diags, identity, kron
from scipy.sparse.linalg import spsolve


def lap_matrix(N):
    h = 1/N
    n = N-1
    T = diags([-1., 2., -1.], [-1, 0, 1], shape=(n, n))
    I = identity(n)
    A = (kron(I, T) + kron(T, I))/h**2
    return A.tocsr()


def poisson(N, f):
    h = 1/N
    x = np.linspace(0, 1, N+1)
    X, Y = np.meshgrid(x, x, indexing='ij')
    A = lap_matrix(N)
    b = f(X[1:-1, 1:-1], Y[1:-1, 1:-1]).ravel()
    u = np.zeros((N+1, N+1))
    u[1:-1, 1:-1] = spsolve(A, b).reshape(N-1, N-1)
    return x, u
