import numpy as np
from scipy.linalg import solve_banded
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


def masks(N):
    i, j = np.indices((N+1, N+1))
    inner = (i > 0) & (i < N) & (j > 0) & (j < N)
    red = inner & ((i+j) % 2 == 0)
    black = inner & ((i+j) % 2 == 1)
    return red, black


def nbr(u):
    s = np.zeros_like(u)
    s[1:-1, 1:-1] = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
    return s


def jacobi(u0, h, tol=1e-6, max_iter=100000, f=None):
    u = u0.copy()
    N = u.shape[0]-1
    src = np.zeros_like(u) if f is None else h**2*f
    inner = np.zeros_like(u, dtype=bool)
    inner[1:-1, 1:-1] = True
    for k in range(1, max_iter+1):
        new = u.copy()
        new[inner] = ((nbr(u) + src)/4)[inner]
        err = np.max(np.abs(new-u))
        u = new
        if err < tol:
            break
    return u, k


def sor(u0, h, w, tol=1e-6, max_iter=100000, f=None):
    u = u0.copy()
    N = u.shape[0]-1
    src = np.zeros_like(u) if f is None else h**2*f
    red, black = masks(N)
    for k in range(1, max_iter+1):
        old = u.copy()
        for m in (red, black):
            gs = (nbr(u) + src)/4
            u[m] = (1-w)*u[m] + w*gs[m]
        err = np.max(np.abs(u-old))
        if err < tol:
            break
    return u, k


def gauss_seidel(u0, h, tol=1e-6, max_iter=100000, f=None):
    return sor(u0, h, 1.0, tol, max_iter, f)


def adi(N, f, rho, tol=1e-6, max_iter=10000):
    h = 1/N
    n = N-1
    x = np.linspace(0, 1, N+1)
    X, Y = np.meshgrid(x, x, indexing='ij')
    F = f(X[1:-1, 1:-1], Y[1:-1, 1:-1])
    ab = np.zeros((3, n))
    ab[0, 1:] = -1/h**2
    ab[1, :] = 2/h**2 + rho
    ab[2, :-1] = -1/h**2
    U = np.zeros((n, n))
    for k in range(1, max_iter+1):
        # y explicit, x implcit
        Hy = (2*U - np.pad(U, ((0, 0), (1, 0)))[:, :-1] - np.pad(U, ((0, 0), (0, 1)))[:, 1:])/h**2
        Us = solve_banded((1, 1), ab, rho*U - Hy + F)
        Hx = (2*Us - np.pad(Us, ((1, 0), (0, 0)))[:-1, :] - np.pad(Us, ((0, 1), (0, 0)))[1:, :])/h**2
        Un = solve_banded((1, 1), ab, (rho*Us - Hx + F).T).T
        err = np.max(np.abs(Un-U))
        U = Un
        if err < tol:
            break
    u = np.zeros((N+1, N+1))
    u[1:-1, 1:-1] = U
    return u, k


def laplace_series(x, y, nterms=100):
    s = np.zeros(np.broadcast(x, y).shape)
    for m in range(nterms):
        n = 2*m+1
        # sinh(n pi y)/sinh(n pi) as exponentials
        r = (np.exp(n*np.pi*(y-1)) - np.exp(-n*np.pi*(y+1)))/(1-np.exp(-2*n*np.pi))
        s += 4/(n*np.pi)*np.sin(n*np.pi*x)*r
    return s
