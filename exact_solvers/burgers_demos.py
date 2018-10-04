"""
Additional functions and demos for Burgers' equation.
"""
from clawpack import pyclaw
from clawpack import riemann
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import sys
sys.path.append('../utils')
from utils import riemann_tools
from . import burgers

def bump_figure(t):
    """Plots bump-into-wave figure at different times for interactive figure."""
    x = np.arange(-11.0,11.0,0.1)
    y = np.exp(-x*x/10)
    x2 = 1.0*x
    x2 = x2 + t*y
    plt.plot(x, y, '--k')
    plt.plot(x2, y, '-k')
    plt.xlim([-10,10])
    if t != 0:
        numarrows = 7
        arrowIndexList = np.linspace(len(x)/3,2*len(x)/3,numarrows, dtype = int)
        for i in arrowIndexList:
            plt.arrow(x[i], y[i], t*y[i]-0.5, 0, head_width=0.02, head_length=0.4, fc='k', ec='k')

def shock():
    """Returns plot function for a shock solution."""
    q_l, q_r = 5.0, 1.0
    states, speeds, reval, wave_type = burgers.exact_riemann_solution(q_l ,q_r)

    plot_function = riemann_tools.make_plot_function(states, speeds, reval, wave_type, 
                                                    layout='horizontal',
                                                    variable_names=['q'],
                                                    plot_chars=[burgers.speed])
    return plot_function

def rarefaction_figure(t):
    """Plots rarefaction figure at different times for interactive figure."""
    numarrows = 6
    x = [-10., 0.0]
    y = [0.2, 0.2]
    for i in range(numarrows):
        x.append(0.0)
        y.append(y[0] + (i+1)*(1.0-y[0])/(numarrows+1))
    x.extend([0.0,10.0])
    y.extend([1.0,1.0])
    x2 = 1.0*np.array(x)
    x2[1:-1] = x2[1:-1] + t*np.array(y[1:-1])
    plt.plot(x, y, '--k')
    plt.plot(x2, y, '-k')
    plt.xlim([-10,10])
    plt.ylim([0.0,1.2])
    if t != 0:
        for i in range(numarrows):
            plt.arrow(x[2+i], y[2+i], np.abs(t*y[2+i]-0.4), 0, head_width=0.02, head_length=0.4, fc='k', ec='k')

def rarefaction():
    """Returns plot function for a rarefaction solution."""
    q_l, q_r = 2.0, 4.0
    states, speeds, reval, wave_type = burgers.exact_riemann_solution(q_l ,q_r)

    plot_function = riemann_tools.make_plot_function(states, speeds, reval, wave_type, 
                                                    layout='horizontal',
                                                    variable_names=['q'],
                                                    plot_chars=[burgers.speed])
    return plot_function

def unphysical():
    """Returns plot function for an unphysical solution."""
    q_l, q_r = 1.0, 5.0
    states, speeds, reval, wave_type = burgers.unphysical_riemann_solution(q_l ,q_r)

    plot_function = riemann_tools.make_plot_function(states, speeds, reval, wave_type, 
                                                    layout='horizontal',
                                                    variable_names=['q'],
                                                    plot_chars=[burgers.speed])
    return plot_function

def bump_animation(numframes):
    """Plots animation of solution with bump initial condition, 
    using pyclaw (calls bump_pyclaw)."""
    x, frames = bump_pyclaw(numframes) 
    fig = plt.figure()
    ax = plt.axes(xlim=(-1, 1), ylim=(-0.2, 1.2))
    line, = ax.plot([], [], '-k', lw=2)

    def fplot(frame_number):
        frame = frames[frame_number]
        pressure = frame.q[0,:]
        line.set_data(x,pressure)
        return line,

    return animation.FuncAnimation(fig, fplot, frames=len(frames), interval=30)

def bump_pyclaw(numframes):
    """Returns pyclaw solution of bump initial condition."""
    # Set pyclaw for burgers equation 1D
    claw = pyclaw.Controller()
    claw.tfinal = 1.5           # Set final time
    claw.keep_copy = True       # Keep solution data in memory for plotting
    claw.output_format = None   # Don't write solution data to file
    claw.num_output_times = numframes  # Number of output frames
    claw.solver = pyclaw.ClawSolver1D(riemann.burgers_1D)  # Choose burgers 1D Riemann solver
    claw.solver.all_bcs = pyclaw.BC.periodic               # Choose periodic BCs
    claw.verbosity = False                                 # Don't print pyclaw output
    domain = pyclaw.Domain( (-1.,), (1.,), (500,))         # Choose domain and mesh resolution
    claw.solution = pyclaw.Solution(claw.solver.num_eqn,domain)
    # Set initial condition
    x=domain.grid.x.centers
    claw.solution.q[0,:] = np.exp(-10 * (x)**2)     
    claw.solver.dt_initial = 1.e99
    # Run pyclaw
    status = claw.run()
    
    return x, claw.frames

def triplestate_animation(ql, qm, qr, numframes):
    """Plots animation of solution with triple-state initial condition, using pyclaw (calls  
    triplestate_pyclaw). Also plots characteristic structure by plotting contour plots of the 
    solution in the x-t plane """
    # Get solution for animation and set plot
    x, frames = triplestate_pyclaw(ql, qm, qr, numframes) 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,4))
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 5)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(0, 2)
    ax1.set_title('Solution q(x)')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$q$')
    ax2.set_title('xt-plane')
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$t$')
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    line1, = ax1.plot([], [], '-k', lw=2)

    # Contour plot of high-res solution to show characteristic structure in xt-plane
    meshpts = 600
    numframes2 = 600
    x2, frames2 = triplestate_pyclaw(ql, qm, qr, numframes2) 
    characs = np.zeros([numframes2,meshpts])
    xx = np.linspace(-3,3,meshpts)
    tt = np.linspace(0,2,numframes2)
    for j in range(numframes2):
        characs[j] = frames2[j].q[0]
    X,T = np.meshgrid(xx,tt)
    ax2.contour(X, T, characs, 38, colors='k')
    # Add animated time line to xt-plane
    line2, = ax2.plot(x, 0*x , '--k')

    line = [line1, line2]

    # Update data function for animation
    def fplot(frame_number):
        frame = frames[frame_number]
        pressure = frame.q[0,:]
        line[0].set_data(x,pressure)
        line[1].set_data(x,0*x+frame.t)
        return line

    return animation.FuncAnimation(fig, fplot, frames=len(frames), interval=30, blit=False)

def triplestate_pyclaw(ql, qm, qr, numframes):
    """Returns pyclaw solution of triple-state initial condition."""
    # Set pyclaw for burgers equation 1D
    meshpts = 600
    claw = pyclaw.Controller()
    claw.tfinal = 2.0           # Set final time
    claw.keep_copy = True       # Keep solution data in memory for plotting
    claw.output_format = None   # Don't write solution data to file
    claw.num_output_times = numframes  # Number of output frames
    claw.solver = pyclaw.ClawSolver1D(riemann.burgers_1D)  # Choose burgers 1D Riemann solver
    claw.solver.all_bcs = pyclaw.BC.extrap               # Choose periodic BCs
    claw.verbosity = False                                # Don't print pyclaw output
    domain = pyclaw.Domain( (-3.,), (3.,), (meshpts,))   # Choose domain and mesh resolution
    claw.solution = pyclaw.Solution(claw.solver.num_eqn,domain)
    # Set initial condition
    x=domain.grid.x.centers
    q0 = 0.0*x
    xtick1 = int(meshpts/3)
    xtick2 = int(2*meshpts/3)
    q0[0:xtick1] = ql
    q0[xtick1:xtick2] = qm
    q0[xtick2:meshpts] = qr
    claw.solution.q[0,:] = q0    
    claw.solver.dt_initial = 1.e99
    # Run pyclaw
    status = claw.run()
    
    return x, claw.frames




