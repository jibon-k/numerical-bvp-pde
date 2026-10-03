# numerical-bvp-pde

Numerical methods for ODE boundary value problems and PDEs (MTH5110). Each problem has an exact solution, so every notebook reports the error.

## Notebooks

1. `01_linear_bvp`: central differences with a banded solve, log-log convergence plot
2. `02_nonlinear_bvp`: finite differences with Newton, residual per iteration and a second solution
3. `03_rayleigh_ritz`: hat function FEM against finite differences
4. `04_poisson`: five-point stencil with a sparse `kron` matrix, spy plot and surface
5. `05_laplace_iterative`: Jacobi, Gauss-Seidel and SOR iteration counts, heatmap
6. `06_adi`: Peaceman-Rachford ADI against the other iterative solvers, runtime
7. `07_heat_explicit`: FTCS stable at r = 0.4 and unstable at r = 0.6
8. `08_crank_nicolson`: stable at r = 5, second order in time and space
9. `09_wave_1d`: CFL condition with sigma = 0.5, 1.0 and 1.1
10. `10_wave_2d`: vibrating membrane (`figures/10_membrane.gif`) and error over time

## Setup

```
uv venv
uv pip install -r requirements.txt
```

Run the notebooks from the `notebooks` folder with `.venv/bin/jupyter lab`.

## Layout

```
notebooks/   01_linear_bvp.ipynb ... 10_wave_2d.ipynb
src/         fd_ode.py, fem_1d.py, elliptic.py, heat.py, wave.py
figures/     saved plots and the membrane gif
```
