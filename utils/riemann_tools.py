"""
This version of riemann_tools.py was adapted from the
version in clawpack/riemann/src from Clawpack V5.3.1 to
add some new features and improve the plots.

This may be modified further as the notebooks in this
directory are improved and expanded.  Eventually a
stable version of this should be moved back to
clawpack/riemann/src in a future release.
"""

from __future__ import absolute_import
from __future__ import print_function
import sympy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from six.moves import range
import copy

try:
    from JSAnimation import IPython_display
except:
    try:
        from clawpack.visclaw.JSAnimation import IPython_display
    except:
        print("*** Warning: JSAnimation not found")
from IPython.display import display


sympy.init_printing(use_latex='mathjax')


def riemann_solution(solver,q_l,q_r,aux_l=None,aux_r=None,t=0.2,problem_data=None,
                     verbose=False,stationary_wave=False,fwave=False):
    r"""
    Compute the (approximate) solution of the Riemann problem with states (q_l, q_r)
    based on the (approximate) Riemann solver `solver`.  The solver should be
    a pointwise solver provided as a Python function.

    **Example**::

        # Call solver
        >>> from clawpack import riemann
        >>> gamma = 1.4
        >>> problem_data = { 'gamma' : gamma, 'gamma1' : gamma - 1 }
        >>> solver = riemann.euler_1D_py.euler_hll_1D
        >>> q_l = (3., -0.5, 2.); q_r = (1., 0., 1.)
        >>> states, speeds, reval = riemann_solution(solver, q_l, q_r, problem_data=problem_data)

        # Check output
        >>> q_m = np.array([1.686068, 0.053321, 1.202282])
        >>> assert np.allclose(q_m, states[:,1])

    """
    if not isinstance(q_l, np.ndarray):
        q_l = np.array(q_l)   # in case q_l, q_r specified as scalars
        q_r = np.array(q_r)

    num_eqn = len(q_l)

    if aux_l is None:
        aux_l = np.zeros(num_eqn)
        num_aux = num_eqn
        aux_r = np.zeros(num_eqn)
    else:
        num_aux = len(aux_l)

    wave, s, amdq, apdq = solver(q_l.reshape((num_eqn,1)),q_r.reshape((num_eqn,1)),
                                 aux_l.reshape((num_aux,1)),aux_r.reshape((num_aux,1)),problem_data)

    num_waves = wave.shape[1]

    if fwave:
        for i in range(num_waves):
            # This will break for zero wave speed
            wave[:,i,:] = wave[:,i,:]/s[i,:]

    if stationary_wave:

        # Allow for unidentified jumps at x=0
        lstates = [q_l]
        for i in range(num_waves):
            if s[i,0] < 0:
                lstates.append(lstates[-1]+np.squeeze(wave[:,i,:]))
            else: break
        rstates = [q_r]
        for j in range(num_waves-1,i-1,-1):
            rstates.insert(0,rstates[0]-np.squeeze(wave[:,j,:]))
        states = np.array(lstates+rstates).T
        s=np.insert(s,i,0)
        jump = rstates[0]-lstates[-1]
        jump = jump.reshape((1,num_eqn,1))
        wave=np.insert(wave,i,jump,axis=1)
        num_waves += 1
    else:
        wave0 = wave[:,:,0]
        num_waves = wave.shape[1]
        qlwave = np.vstack((q_l,wave0.T)).T
        # Sum to the waves to get the states:
        states = np.cumsum(qlwave,1)
        if not np.allclose(states[:,-1],q_r):
            print("""Warning: Computed right state does not match input.
                   You may need to set stationary_wave=True.""")

    num_states = num_waves + 1

    if verbose:
        print('States in Riemann solution:')
        states_sym = sympy.Matrix(states)
        display([states_sym[:,k] for k in range(num_states)])

        print('Waves (jumps between states):')
        wave_sym = sympy.Matrix(wave[:,:,0])
        display([wave_sym[:,k] for k in range(num_waves)])

        print("Speeds: ")
        s_sym = sympy.Matrix(s)
        display(s_sym.T)

        print("Fluctuations amdq, apdq: ")
        amdq_sym = sympy.Matrix(amdq).T
        apdq_sym = sympy.Matrix(apdq).T
        display([amdq_sym, apdq_sym])

    def riemann_eval(xi):
        "Return Riemann solution as function of xi = x/t."
        qout = np.zeros((num_eqn,len(xi)))
        intervals = [(xi>=s[i])*(xi<=s[i+1]) for i in range(len(s)-1)]
        intervals.insert(0, xi<s[0])
        intervals.append(xi>=s[-1])
        for m in range(num_eqn):
            qout[m,:] = np.piecewise(xi, intervals, states[m,:])

        return qout

    return states, s, riemann_eval


def plot_phase(states, i_h=0, i_v=1, ax=None, label_h=None, label_v=None):
    """
    Plot 2d phase space plot.
    If num_eqns > 2, can specify which component of q to put on horizontal
    and vertical axes via i_h and i_v.
    """

    if label_h is None:
        label_h = 'q[%s]' % i_h
    if label_v is None:
        label_v = 'q[%s]' % i_v

    q0 = states[i_h,:]
    q1 = states[i_v,:]

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(q0,q1,'o-k')
    #ax.set_title('phase space: %s -- %s' % (label_h,label_v))
    ax.set_title('States in phase space')
    ax.axis('equal')
    dq0 = q0.max() - q0.min()
    dq1 = q1.max() - q1.min()
    ax.text(q0[0] + 0.05*dq0,q1[0] + 0.05*dq1,'q_left')
    ax.text(q0[-1] + 0.05*dq0,q1[-1] + 0.05*dq1,'q_right')
    ax.axis([q0.min()-0.1*dq0, q0.max()+0.1*dq0, q1.min()-0.1*dq1, q1.max()+0.1*dq1])
    ax.set_xlabel(label_h)
    ax.set_ylabel(label_v)

def plot_phase_3d(states):
    """
    3d phase space plot
    """
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(states[0,:],states[1,:],states[2,:],'ko-')
    ax.set_xlabel('q[0]')
    ax.set_ylabel('q[1]')
    ax.set_zlabel('q[2]')
    ax.set_title('phase space')
    ax.text(states[0,0]+0.05,states[1,0],states[2,0],'q_left')
    ax.text(states[0,-1]+0.05,states[1,-1],states[2,-1],'q_right')


def plot_riemann(states, s, riemann_eval, wave_types=None, t=0.1, fig=None, color='multi', layout='horizontal',conserved_variables=None,t_pointer=True, extra_axes=False):
    """
    Take an array of states and speeds s and plot the solution at time t.
    For rarefaction waves, the corresponding entry in s should be tuple of two values,
    which are the wave speeds that bound the rarefaction fan.

    Plots in the x-t plane and also produces a separate plot for each component of q.
    """
    num_eqn,num_states = states.shape
    colors = {}

    if wave_types is None:
        wave_types = ['contact']*len(s)

    if color == 'multi':
        colors['shock'] = 'r'
        colors['raref'] = 'b'
        colors['contact'] = 'k'
    else:
        colors['shock'] = color
        colors['raref'] = color
        colors['contact'] = color

    if fig is None:
        num_axes = num_eqn+1
        if extra_axes: num_axes += 1
        if layout == 'horizontal':
            fig_width = 4*num_axes
            fig, ax = plt.subplots(1,num_axes,figsize=(fig_width,4))
        elif layout == 'vertical':
            fig_width = 9
            fig_height = 4*(num_axes-1)
            fig, ax = plt.subplots(num_axes,1,figsize=(fig_width,fig_height),sharex=True)
            plt.subplots_adjust(hspace=0)
            ax[-1].set_xlabel('x')
            ax[0].set_ylabel('t')
            ax[0].set_title('t = %6.3f' % t)
    else:
        ax = fig.axes
        xmin, xmax = ax[1].get_xlim()

    for axis in ax:
        for child in axis.get_children():
            if isinstance(child, matplotlib.spines.Spine):
                child.set_color('#dddddd')

    tmax = 1.0
    xmax = 0.
    for i in range(len(s)):
        if wave_types[i] in ['shock','contact']:
            x1 = tmax * s[i]
            ax[0].plot([0,x1],[0,tmax],color=colors[wave_types[i]])
            xmax = max(xmax,abs(x1))
        else:  # plot rarefaction fan
            speeds = np.linspace(s[i][0],s[i][1],5)
            for ss in speeds:
                x1 = tmax * ss
                ax[0].plot([0,x1],[0,tmax],color=colors['raref'],lw=0.3)
                xmax = max(xmax,abs(x1))

    x = np.linspace(-xmax,xmax,1000)

    ax[0].set_xlim(-xmax,xmax)
    ax[0].plot([-xmax,xmax],[t,t],'--k',linewidth=0.5)
    if t_pointer:
        ax[0].text(-1.8*xmax,t,'t = %4.2f -->' % t)
    ax[0].set_title('Waves in x-t plane')

    if conserved_variables is None:
        conserved_variables = ['q[%s]' % i for i in range(num_eqn)]

    for i in range(num_eqn):
        ax[i+1].set_xlim((-1,1))
        qmax = states[i,:].max()  # max([state[i] for state in states])
        qmin = states[i,:].min()  # min([state[i] for state in states])
        qdiff = qmax - qmin
        ax[i+1].set_xlim(-xmax,xmax)
        ax[i+1].set_ylim((qmin-0.1*qdiff,qmax+0.1*qdiff))
        if layout == 'horizontal':
            ax[i+1].set_title(conserved_variables[i]+' at t = %6.3f' % t)
        elif layout == 'vertical':
            ax[i+1].set_ylabel(conserved_variables[i])

    if t == 0:
        q = riemann_eval(x/1e-10)
    else:
        q = riemann_eval(x/t)

    for i in range(num_eqn):
        ax[i+1].plot(x,q[i][:],color=colors['contact'],lw=2)

    return fig


def make_plot_function(states_list,speeds_list,riemann_eval_list,
                       wave_types_list=None,names=None,layout='horizontal',
                       conserved_variables=None,colors=('multi','green','orange'),
                       plot_chars=False):
    """
    Utility function to create a plot_function that takes a single argument t.
    This function can then be used with ipywidgets.interact.
    Version that takes an arbitrary list of sets of states and speeds in order to make a comparison.

    plot_chars: If provided, ordinary characteristics are included in the x-t plot.
                The value of this argument should be a function c(q,x) that gives
                the characteristic speed.
    """

    if type(states_list) is not list:
        states_list = [states_list]
        speeds_list = [speeds_list]
        riemann_eval_list = [riemann_eval_list]
        if wave_types_list is not None:
            wave_types_list = [wave_types_list]

    if wave_types_list is None:
        wave_types_list= [['contact']*len(speeds_list[0])]*len(speeds_list)

    def plot_function(t):
        fig = None
        for i in range(len(states_list)):
            states = states_list[i]
            speeds = speeds_list[i]
            riemann_eval = riemann_eval_list[i]
            wave_types = wave_types_list[i]

            if fig is None:
                fig = plot_riemann(states,speeds,riemann_eval,wave_types,t,color=colors[i],layout=layout,conserved_variables=conserved_variables)
            else:
                fig = plot_riemann(states,speeds,riemann_eval,wave_types,t,fig,colors[i],layout=layout,conserved_variables=conserved_variables,t_pointer=False)

            if names is not None:
                # We could use fig.legend here if we had the line plot handles
                fig.axes[1].legend(names,loc='best')

            if plot_chars:
                plot_characteristics(states,speeds,plot_chars,fig.axes[0])

        plt.show()
        return None

    return plot_function

def JSAnimate_plot_riemann(states,speeds,riemann_eval, wave_types=None, times=None, **kwargs):
    try:
        from utils import animation_tools
    except:
        try:
            from clawpack.visclaw import animation_tools
        except:
            print("*** Warning: animation_tools not found")

    figs = []  # to collect figures at multiple times
    if times is None:
        times = np.linspace(0,0.9,10)
    for t in times:
        fig = plot_riemann(states,speeds,riemann_eval,wave_types,t=t, **kwargs)
        figs.append(fig)
        plt.close(fig)

    images = animation_tools.make_images(figs)
    anim = animation_tools.JSAnimate_images(images, figsize=(10,5))
    return anim


def plot_riemann_trajectories(states, s, riemann_eval, wave_types=None,
                              i_vel=1, fig=None, color='b', num_left=10,
                              num_right=10, xmax=None, rho_left=None,
                              rho_right=None):
    """
    Take an array of states and speeds s and plot the solution in the x-t plane,
    along with particle trajectories.

    Only useful for systems where one component is velocity.
    i_vel should be the index of this component.

    For rarefaction waves, the corresponding entry in s should be tuple of two values,
    which are the wave speeds that bound the rarefaction fan.

    """
    colors = {'shock': 'r', 'raref': 'b', 'contact': 'k'}
    if wave_types is None:
        wave_types = ['contact']*len(s)

    num_eqn,num_states = states.shape
    if fig is None:
        fig, ax = plt.subplots()

    # auto scale the x axis?
    xmax_auto = (xmax is None)

    tmax = 1.0
    if xmax_auto:
        xmax = 0.
    for i in range(len(s)):
        if wave_types[i] in ['shock','contact']:
            x1 = tmax * s[i]
            ax.plot([0,x1],[0,tmax],color=colors[wave_types[i]])
            if xmax_auto:
                xmax = max(xmax,abs(x1))
        else:  # plot rarefaction fan
            speeds = np.linspace(s[i][0],s[i][1],5)
            for ss in speeds:
                x1 = tmax * ss
                ax.plot([0,x1],[0,tmax],color=colors['raref'],lw=0.5)
                if xmax_auto:
                    xmax = max(xmax,abs(x1))

    ax.set_xlim(-xmax,xmax)

    xx_left = np.linspace(-xmax,0,num_left)
    xx_right = np.linspace(0,xmax,num_right)

    # instead define spacing based on density if available:
    if rho_left is not None:
        xx_left = np.arange(-xmax, 0, xmax*0.02/(rho_left + 1e-8))
    if rho_right is not None:
        xx_right = np.arange(0, xmax, xmax*0.02/(rho_right + 1e-8))

    xx = np.hstack((xx_left, xx_right))
    xtraj = [xx]

    nsteps = 200.
    dt = 1./nsteps
    tt = np.linspace(0,1,nsteps+1)
    q_old = riemann_eval(xx/1e-15)
    for n in range(1,len(tt)):
        q_new = riemann_eval(xx/tt[n])
        v_mid = 0.5*(q_old[i_vel,:] + q_new[i_vel,:])
        xx = xx + dt*v_mid
        xtraj.append(xx)
        q_old = copy.copy(q_new)

    xtraj = np.array(xtraj)
    for j in range(xtraj.shape[1]):
        plt.plot(xtraj[:,j],tt,'k')

    ax.set_title('Waves and particle trajectories in x-t plane')
    plt.show()

def plot_characteristics(states, speeds, char_speed, ax):
    """
    Plot characteristics in constant regions.

    c: Function c(q,xi) that gives the characteristic speed.
    """

    # Find constant state regions, in terms of xi = x/t
    constant_regions = []
    current = -1.e3
    for s in speeds:
        if type(s) in (tuple, list):
            constant_regions.append((current,s[0]))
            current = s[1]
        else:
            constant_regions.append((current,s))
            current = s
    constant_regions.append((current,1.e3))
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(0,1)
    xmin, xmax, tmin, tmax = ax.axis()

    for i,region in enumerate(constant_regions):
        xi_m = region[0]  # characteristic bounding this constant region on the left
        xi_p = region[1]  # characteristic bounding this constant region on the right
        if xi_m == xi_p: continue
        if xi_m == -np.Inf:
            xi_bar = xi_p*2.
        elif xi_p == np.Inf:
            xi_bar = xi_m*2.
        else:
            xi_bar = (xi_m+xi_p)/2.
        c = char_speed(states[0][i], xi_bar)

        if xi_bar < 0:
            x_bar = max(xmin, xi_bar*tmax)
        else:
            x_bar = min(xmax, xi_bar*tmax)

        for x0 in np.linspace(0, x_bar, 7)[1:-1]:
            t0 = x0/xi_bar
            # Plot characteristic passing through (x0, t0)

            # Find intersection of this characteristic with bounding rays
            if c == xi_m:
                x_m = -np.Inf
            else:
                x_m = (x0-c*t0)/(1-c/xi_m)
            if c == xi_p:
                x_p = np.Inf
            else:
                x_p = (x0-c*t0)/(1-c/xi_p)
            # Find intersection with bottom and top of axes:
            x_bottom = x0 + c*(tmin - t0)
            x_top = x0 + c*(tmax - t0)

            if c < 0:
                x_left = max(xmin, x_top)
                x_right = min(xmax, x_bottom)
            else:
                x_left = max(xmin, x_bottom)
                x_right = min(xmax, x_top)

            if x_m < x0: x_left = max(x_m, x_left)
            if x_p < x0: x_left = max(x_p, x_left)
            if x_m > x0: x_right = min(x_m, x_right)
            if x_p > x0: x_right = min(x_p, x_right)

            t = lambda x: (x - x0)/c + t0
            ax.plot([x_left, x_right], [t(x_left), t(x_right)],'-k',linewidth=0.2)

        ax.set_xlim(-0.5,0.5)
        ax.set_ylim(0,1)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
