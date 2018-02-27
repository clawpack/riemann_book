"""
Exact Riemann solvers for Burgers' equation in 1D and interactive plot function.
"""
import numpy as np
from ipywidgets import widgets, interact
from IPython.display import display


def speed(q, xi):
    "Characteristic speed."
    return q

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


def plot_interactive_riemann(plot_riemann):
    ql_widget = widgets.FloatSlider(min=0., max=1., value=0.5, description=r'$q_l$')
    qr_widget = widgets.FloatSlider(min=0., max=1., value=0.0, description=r'$q_r$')
    t_widget = widgets.FloatSlider(min=0.,max=1.,value=0.1, description=r'$t$')
    x_range_widget = widgets.FloatSlider(min=0.1,max=5,value=1.,description=r'x-axis range')
    interact_gui = widgets.HBox([t_widget,widgets.VBox([ql_widget,qr_widget]),x_range_widget])

    burgers_widget = interact(plot_riemann, q_l=ql_widget, q_r=qr_widget, 
                              t=t_widget, x_range=x_range_widget)
    burgers_widget.widget.close()
    display(interact_gui)
    display(burgers_widget.widget.out)


