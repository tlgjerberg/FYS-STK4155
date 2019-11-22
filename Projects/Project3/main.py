import numpy as np
from pdesolvers import *

L = 1
T = 1

beta = 0.4
dx = 1. / 10
dt = dx * beta

Nx = int(L / dx)
Nt = int(T / dt)

x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)

pde_obj = PDEsolvers()

pde_obj.initial_condition(x)

u = pde_obj.ForwardEuler(Nx, Nt, dt, dx)
print(u)

print(f"The domain is: T = {T} with {Nt} points and L = {L} with {Nx} points")
print(
    f"Stability coefficient is Beta = {beta}, with dx = {dx}, and dt = {dt}")
