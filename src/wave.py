import numpy as np


def wave_1d(N, sigma, T):
    h = 1/N
    dt = sigma*h
    nt = int(round(T/dt))
    x = np.linspace(0, 1, N+1)
    u0 = np.sin(np.pi*x)
    u1 = u0.copy()
    u1[1:-1] = u0[1:-1] + sigma**2/2*(u0[:-2]-2*u0[1:-1]+u0[2:])
    err = [0, np.max(np.abs(u1 - np.cos(np.pi*dt)*u0))]
    mx = [np.max(np.abs(u0)), np.max(np.abs(u1))]
    for k in range(1, nt):
        u2 = np.zeros_like(u0)
        u2[1:-1] = 2*u1[1:-1] - u0[1:-1] + sigma**2*(u1[:-2]-2*u1[1:-1]+u1[2:])
        u0, u1 = u1, u2
        t = (k+1)*dt
        err.append(np.max(np.abs(u1 - np.cos(np.pi*t)*np.sin(np.pi*x))))
        mx.append(np.max(np.abs(u1)))
    t = nt*dt
    return {'x': x, 'u': u1, 't': t, 'err': np.array(err), 'mx': np.array(mx), 'dt': dt}
