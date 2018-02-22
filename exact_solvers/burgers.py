"""
Exact Riemann solvers for Burgers' equation in 1D.
"""
import numpy as np

def lambda1(q, xi, aux):
    "Characteristic speed."
    rho, bulk = aux
    return -np.sqrt(bulk/rho)

def exact_riemann_solution(q_l,q_r):
    r"""Exact solution to the Riemann problem for the LWR traffic model."""
    f = lambda q: 0.5*q*q
    states = np.array([[q_l, q_r]])
    if q_l > q_r:  # Shock wave
        shock_speed = (f(q_l)-f(q_r))/(q_l-q_r)
        speeds = [shock_speed]
        wave_types = ['shock']
        def reval(xi):
            q = np.zeros((1,len(xi)))
            q[0,:] = (xi < shock_speed)*q_l \
              + (xi >=shock_speed)*q_r
            return q

    else:  # Rarefaction wave
        c_l  = q_l
        c_r = q_r

        speeds = [[c_l,c_r]]
        wave_types = ['rarefaction']

        def reval(xi):
            q = np.zeros((1,len(xi)))
            q[0,:] = (xi<=c_l)*q_l \
              + (xi>=c_r)*q_r \
              + (c_l<xi)*(xi<c_r)*xi
            return q

    return states, speeds, reval, wave_types




