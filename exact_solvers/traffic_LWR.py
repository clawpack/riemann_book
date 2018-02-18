import numpy as np
import sys
sys.path.append('../utils')
from utils import riemann_tools

def riemann_traffic_exact(q_l,q_r):
    r"""Exact solution to the Riemann problem for the LWR traffic model."""
    f = lambda q: q*(1-q)
    states = np.array([[q_l, q_r]])
    if q_r > q_l:  # Shock wave
        shock_speed = (f(q_l)-f(q_r))/(q_l-q_r)
        speeds = [shock_speed]
        wave_types = ['shock']
        def reval(xi):
            q = np.zeros((1,len(xi)))
            q[0,:] = (xi < shock_speed)*q_l \
              + (xi >=shock_speed)*q_r
            return q

    else:  # Rarefaction wave
        c_l  = 1-2*q_l
        c_r = 1-2*q_r

        speeds = [[c_l,c_r]]
        wave_types = ['rarefaction']

        def reval(xi):
            q = np.zeros((1,len(xi)))
            q[0,:] = (xi<=c_l)*q_l \
              + (xi>=c_r)*q_r \
              + (c_l<xi)*(xi<c_r)*(1.-xi)/2.
            return q

    return states, speeds, reval, wave_types

def plot_car_trajectories(q_l,q_r,ax=None,t=None,xmax=None):
    states, speeds, reval, wave_types = riemann_traffic_exact(q_l,q_r)
    def reval_with_speed(xi):
        q = reval(xi)
        u = 1-q
        qu = np.vstack((q,u))
        return qu

    # density of particles for trajectories:
    rho_left = q_l / 3.
    rho_right = q_r / 3.

    # compute trajectories:
    x_traj, t_traj, xmax = riemann_tools.compute_riemann_trajectories(states, 
            speeds, reval_with_speed, wave_types,
            xmax=xmax,rho_left=rho_left, rho_right=rho_right)

    # plot trajectories along with waves in the x-t plane:
    riemann_tools.plot_riemann_trajectories(x_traj, t_traj, speeds, wave_types, 
            xmax=xmax, ax=ax, t=t)
