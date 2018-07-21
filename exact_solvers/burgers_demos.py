from clawpack import pyclaw
from clawpack import riemann
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import sys
sys.path.append('../utils')
from utils import riemann_tools
from . import burgers

def shock():
    q_l, q_r = 5.0, 1.0
    states, speeds, reval, wave_type = burgers.exact_riemann_solution(q_l ,q_r)

    plot_function = riemann_tools.make_plot_function(states, speeds, reval, wave_type, 
                                                    layout='horizontal',
                                                    variable_names=['q'],
                                                    plot_chars=[burgers.speed])
    return plot_function

def rarefaction():
    q_l, q_r = 2.0, 4.0
    states, speeds, reval, wave_type = burgers.exact_riemann_solution(q_l ,q_r)

    plot_function = riemann_tools.make_plot_function(states, speeds, reval, wave_type, 
                                                    layout='horizontal',
                                                    variable_names=['q'],
                                                    plot_chars=[burgers.speed])
    return plot_function

def unphysical():
    q_l, q_r = 1.0, 5.0
    states, speeds, reval, wave_type = burgers.unphysical_riemann_solution(q_l ,q_r)

    plot_function = riemann_tools.make_plot_function(states, speeds, reval, wave_type, 
                                                    layout='horizontal',
                                                    variable_names=['q'],
                                                    plot_chars=[burgers.speed])
    return plot_function

def bump_animation(numframes):
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

def triplestate_animation(ql, qm, qr, numframes):
    x, frames = triplestate_pyclaw(ql, qm, qr, numframes) 
    fig = plt.figure()
    ax = plt.axes(xlim=(-3, 3), ylim=(-2, 5))
    line, = ax.plot([], [], '-k', lw=2)

    def fplot(frame_number):
        frame = frames[frame_number]
        pressure = frame.q[0,:]
        line.set_data(x,pressure)
        return line,

    return animation.FuncAnimation(fig, fplot, frames=len(frames), interval=30)

def bump_pyclaw(numframes):
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

def triplestate_pyclaw(ql, qm, qr, numframes):
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
	




