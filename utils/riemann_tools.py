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


def plot_waves(states, s, riemann_eval, wave_types, t=0.1, ax=None, color='multi', t_pointer=True):

    num_eqn,num_states = states.shape

    if wave_types is None:
        wave_types = ['contact']*len(s)

    colors = {}
    if color == 'multi':
        colors['shock'] = 'r'
        colors['raref'] = 'b'
        colors['contact'] = 'k'
    else:
        colors['shock'] = color
        colors['raref'] = color
        colors['contact'] = color

    if ax is None:
        fig, ax = plt.subplots()

    tmax = 1.0
    xmax = 0.
    for i in range(len(s)):
        if wave_types[i] in ['shock','contact']:
            x1 = tmax * s[i]
            ax.plot([0,x1],[0,tmax],color=colors[wave_types[i]])
            xmax = max(xmax,abs(x1))
        else:  # plot rarefaction fan
            speeds = np.linspace(s[i][0],s[i][1],5)
            for ss in speeds:
                x1 = tmax * ss
                ax.plot([0,x1],[0,tmax],color=colors['raref'],lw=0.6)
                xmax = max(xmax,abs(x1))

    ax.set_xlim(-xmax,xmax)
    ax.plot([-xmax,xmax],[t,t],'--k',linewidth=0.5)
    if t_pointer:
        ax.text(-1.8*xmax,t,'t = %4.2f -->' % t)
    ax.set_title('Waves in x-t plane')
    ax.set_ylim(0,tmax)


def plot_riemann(states, s, riemann_eval, wave_types=None, t=0.1, ax=None,
                 color='multi', layout='horizontal', variable_names=None,
                 t_pointer=True, extra_axes=False, fill=(),
                 derived_variables=None):
    """
    Take an array of states and speeds s and plot the solution at time t.
    For rarefaction waves, the corresponding entry in s should be tuple of two values,
    which are the wave speeds that bound the rarefaction fan.

    Plots in the x-t plane and also produces a separate plot for each component of q.

    derived_variables (optional): A function that takes the conserved variables
                                  and returns the ones to plot.  The number of
                                  plotted variables need not be the same as the
                                  number of conserved variables.
    """
    num_vars, num_states = states.shape
    if derived_variables:
        num_vars = len(derived_variables(states[:,0]))
        for i in range(num_states):
            states[:,i] = derived_variables(states[:,i])
    if ax is not None:
        assert len(ax) == num_vars + 1 + extra_axes

    if wave_types is None:
        wave_types = ['contact']*len(s)

    if ax is None:
        num_axes = num_vars+1
        if extra_axes: num_axes += 1
        if layout == 'horizontal':
            fig_width = 4*num_axes
            fig, ax = plt.subplots(1,num_axes,figsize=(fig_width,4))
            plt.subplots_adjust(wspace=0.5)
        elif layout == 'vertical':
            fig_width = 9
            fig_height = 4*(num_axes-1)
            fig, ax = plt.subplots(num_axes,1,figsize=(fig_width,fig_height),sharex=True)
            plt.subplots_adjust(hspace=0)
            ax[-1].set_xlabel('x')
            ax[0].set_ylabel('t')
            ax[0].set_title('t = %6.3f' % t)
    else:
        xmin, xmax = ax[1].get_xlim()

    for axis in ax:
        for child in axis.get_children():
            if isinstance(child, matplotlib.spines.Spine):
                child.set_color('#dddddd')

    # Plot waves in x-t plane
    plot_waves(states, s, riemann_eval, wave_types, t=t, ax=ax[0], color=color, t_pointer=t_pointer)
    xmax = ax[0].get_xlim()[1]

    # Plot conserved quantities as function of x for fixed t
    if variable_names is None:
        variable_names = ['q[%s]' % i for i in range(num_vars)]

    q_sample = riemann_eval(np.linspace(-10,10))
    if derived_variables:
        q_sample = derived_variables(q_sample)

    for i in range(num_vars):
        ax[i+1].set_xlim((-1,1))
        qmax = q_sample[i][:].max()
        qmin = q_sample[i][:].min()
        qdiff = qmax - qmin
        ax[i+1].set_xlim(-xmax,xmax)
        ax[i+1].set_ylim((qmin-0.1*qdiff,qmax+0.1*qdiff))
        if layout == 'horizontal':
            ax[i+1].set_title(variable_names[i]+' at t = %6.3f' % t)
        elif layout == 'vertical':
            ax[i+1].set_ylabel(variable_names[i])

    x = np.linspace(-xmax,xmax,1000)

    if t == 0:
        q = riemann_eval(x/1e-10)
    else:
        q = riemann_eval(x/t)

    if derived_variables:
        q = derived_variables(q)

    for i in range(num_vars):
        ax[i+1].plot(x,q[i][:],'-k',lw=2)
        if i in fill:
            ax[i+1].fill_between(x,q[i][:],color='b')
            ax[i+1].set_ybound(lower=0)

    return ax


def make_plot_function(states_list,speeds_list,riemann_eval_list,
                       wave_types_list=None,names=None,layout='horizontal',
                       variable_names=None,colors=('multi','green','orange'),
                       plot_chars=None,derived_variables=None):
    """
    Utility function to create a plot_function that takes a single argument t,
    or (if plot_chars is specified) an argument t and an integer argument indicating
    which characteristic family to plot.
    This function can then be used with ipywidgets.interact.

    plot_chars: If provided, ordinary characteristics are included in the x-t plot.
                The value of this argument should be a list of functions c(q,x) that give
                the characteristic speeds of the wave families (in order).
    """
    if type(states_list) is not list:
        states_list = [states_list]
        speeds_list = [speeds_list]
        riemann_eval_list = [riemann_eval_list]
        if wave_types_list is not None:
            wave_types_list = [wave_types_list]

    if wave_types_list is None:
        wave_types_list= [['contact']*len(speeds_list[0])]*len(speeds_list)

    num_eqn,num_states = states_list[0].shape

    num_axes = num_eqn + 1

    def plot_function(t,which_char=None):
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

        for i in range(len(states_list)):
            states = states_list[i]
            speeds = speeds_list[i]
            riemann_eval = riemann_eval_list[i]
            wave_types = wave_types_list[i]

            plot_riemann(states, speeds, riemann_eval, wave_types, t, ax,
                         colors[i], layout=layout,
                         variable_names=variable_names, t_pointer=False,
                         derived_variables=derived_variables)

            if names is not None:
                # We could use fig.legend here if we had the line plot handles
                ax[1].legend(names,loc='best')

            if which_char:
                plot_characteristics(riemann_eval,plot_chars[which_char-1],ax[0])

        plt.show()
        return None

    if plot_chars:
        return plot_function
    else:
        def real_plot_function(t):
            plot_function(t,None)

        return real_plot_function

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
        ax = plot_riemann(states,speeds,riemann_eval,wave_types,t=t, **kwargs)
        figs.append(ax[0].figure)
        plt.close(ax[0].figure)

    images = animation_tools.make_images(figs)
    anim = animation_tools.JSAnimate_images(images, figsize=(10,5))
    return anim


def plot_riemann_trajectories(states, s, riemann_eval, wave_types=None,
                              i_vel=1, fig=None, color='b', num_left=10,
                              num_right=10, xmax=None, rho_left=None,
                              rho_right=None,ax=None):
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
    if ax is None:
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

def plot_characteristics(reval, char_speed, axes=None, extra_lines=None):
    """
    Plot characteristics in x-t plane by integration.

    char_speed: Function char_speed(q,xi) that gives the characteristic speed.
    axes: matplotlib axes on which to plot
    extra_lines: tuple of pairs of pairs; each entry specifies the endpoints of a line
                    along which to start more characteristics
    """
    if axes:
        xmin, xmax, tmin, tmax = axes.axis()
    else:
        xmin, xmax, tmin, tmax = (-1., 1., 0., 0.5)

    Dx = xmax-xmin
    x = np.linspace(xmin-Dx, xmax+Dx, 60)
    t = np.linspace(tmin,tmax,500)
    chars = np.zeros((len(x),len(t)))  # x-t coordinates of characteristics, one curve per row
    chars[:,0] = x
    dt = t[1]-t[0]
    c = np.zeros(len(x))
    for i in range(1,len(t)):
        xi = chars[:,i-1]/max(t[i-1],dt)
        q = np.array(reval(xi))
        for j in range(len(x)):
            c[j] = char_speed(q[:,j],xi[j])
        chars[:,i] = chars[:,i-1] + dt*c  # Euler's method

    for j in range(len(x)):
        axes.plot(chars[j,:],t,'-k',linewidth=0.2,zorder=0)

    if extra_lines:
        for endpoints in extra_lines:
            begin, end = endpoints
            x = np.linspace(begin[0], end[0], 10)
            tstart = np.linspace(begin[1], end[1], 10)
            for epsilon in (-1.e-3, 1.e-3):
                for xx, tt in zip(x+epsilon,tstart):
                    t = np.linspace(tt,tmax,200)
                    dt = t[1]-t[0]
                    char = np.zeros(len(t))
                    char[0] = xx
                    for i in range(1,len(t)):
                        xi = char[i-1]/max(t[i-1],dt)
                        q = np.array(reval(np.array([xi])))
                        c = char_speed(q,xi)
                        char[i] = char[i-1] + dt*c
                    axes.plot(char,t,'-k',linewidth=0.2,zorder=0)

    axes.axis((xmin, xmax, tmin, tmax))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
