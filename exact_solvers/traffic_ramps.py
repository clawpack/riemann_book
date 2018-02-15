import numpy as np
import sys
sys.path.append('../utils')


def riemann_traffic_exact(q_l,q_r,D):
    r"""Exact solution to the Riemann problem for the LWR traffic model
        with an on-ramp at x=0 that brings a flux D of cars.
    """
    assert D <= 0.25  # Maximum flux

    f = lambda q: q*(1-q)
    c = lambda q: 1 - 2*q
    shock_speed = lambda qr, ql: (f(qr)-f(ql))/(qr-ql)

    c_r = c(q_r)

    if q_l<=0.5 and f(q_l) + D < 0.25:
        # No left-going shock
        q_star = 0.5 - np.sqrt(1.-4.*(f(q_l)+D))/2.
        c_star = c(q_star)
        if q_r <= 0.5:
            # Right-going rarefaction
            states = [q_l, q_star, q_r]
            wave_types = ['contact', 'raref']
            speeds = [0,(c_star,c_r)]

            def reval(xi):
                q = np.zeros((1,len(xi)))
                q[0,:] = (xi<=0)*q_l \
                  + (xi>0)*(xi<=c_star)*q_star \
                  + (xi>c_star)*(xi<=c_r)*(1.-xi)/2. \
                  + (xi>c_r)*q_r
                return q

        else:
            # Right-going shock
            s_star = shock_speed(q_star,q_r)
            speeds = [0, s_star]
            states = [q_l, q_star, q_r]
            wave_types = ['contact', 'shock']

            def reval(xi):
                q = np.zeros((1,len(xi)))
                q[0,:] = (xi<=0)*q_l \
                  + (xi>0)*(xi<=s_star)*q_star \
                  + (xi>s_star)*q_r
                return q

    else:
        # Left-going shock
        if q_r <= 0.5:
            q_star = 0.5 + np.sqrt(D)
            states = [q_l, q_star, q_r]
            wave_types = ['shock','contact','raref']
            s_star = shock_speed(q_star,q_l)
            speeds = [s_star,0,(0,c(q_r))]

            def reval(xi):
                q = np.zeros((1,len(xi)))
                q[0,:] = (xi<=s_star)*q_l \
                  + (xi>s_star)*(xi<=0)*q_star \
                  + (xi>0)*(xi<=c_r)*(1.-xi)/2. \
                  + (xi>c_r)*q_r
                return q

        else:
            q_star = 0.5*(1 + np.sqrt(1.+4*D-4*f(q_r)))
            states = [q_l, q_star, q_r]
            wave_types = ['shock','contact']
            s_star = shock_speed(q_star,q_l)
            speeds = [s_star,0]

            def reval(xi):
                q = np.zeros((1,len(xi)))
                q[0,:] = (xi<=s_star)*q_l \
                  + (xi>s_star)*(xi<=0)*q_star \
                  + (xi>0)*q_r
                return q


    return np.array([states]), speeds, reval, wave_types
