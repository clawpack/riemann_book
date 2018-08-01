"""
Additional functions and demos for acoustics equations.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from ipywidgets import widgets, interact
from IPython.display import display
import sys
sys.path.append('../utils')
from utils import riemann_tools
from . import acoustics
colors = ['g','orange']

def decompose_q_interactive():
    """plots interactive decomposed eigenvectors."""
    pwidget = widgets.FloatSlider(min=-1,max=1,value=1.)
    uwidget = widgets.FloatSlider(min=-1,max=1,value=0.)
    rhowidget = widgets.FloatSlider(min=0.1,max=2,value=1.,description=r'$\rho$')
    Kwidget = widgets.FloatSlider(min=0.1,max=2,value=1.)

    interact_gui = widgets.VBox([widgets.HBox([pwidget, rhowidget]),
                                   widgets.HBox([uwidget, Kwidget])]);

    mainwidget = interact(decompose_q, p=pwidget, u=uwidget, rho=rhowidget, K=Kwidget);

    mainwidget.widget.close()
    display(interact_gui)
    display(mainwidget.widget.out)

def decompose_q(p,u,K,rho):
    r"""Plotting function for decompose_q_interactive. It 
    should also print the eigenvectors and the values w_1, w_2."""
    Z = np.sqrt(K*rho)
    fig, axes = plt.subplots(1,2,figsize=(10,6))
    axes[0].arrow(0,0,-Z,1,head_width=0.05, head_length=0.1, color=colors[0])
    axes[0].arrow(0,0,Z,1, head_width=0.05, head_length=0.1, color=colors[1])
    l1 = axes[0].plot([],[],colors[0])
    l2 = axes[0].plot([],[],'-',color=colors[1])
    axes[0].set_xlim(-2,2)
    axes[0].set_ylim(-2,2)
    axes[0].set_aspect('equal')
    axes[0].set_title('Eigenvectors')
    axes[0].legend(['$r_1$','$r_2$'],loc=3)
    axes[0].plot([0,0],[-2,2],'--k',alpha=0.2)
    axes[0].plot([-2,2],[0,0],'--k',alpha=0.2)

    
    axes[1].plot([0,p],[0,u],'k',lw=3)    
    alpha1 = (Z*u-p)/(2.*Z)
    alpha2 = (Z*u+p)/(2.*Z)
    axes[1].plot([0,-Z*alpha1],[0,1*alpha1], color=colors[0], lw=3)
    axes[1].plot([-Z*alpha1,-Z*alpha1+Z*alpha2],[1*alpha1,alpha1+alpha2], color=colors[1], lw=3)
    axes[1].set_xlim(-1.2,1.2)
    axes[1].set_ylim(-1.2,1.2)
    axes[1].set_aspect('equal')
    axes[1].legend(['$w$',r'$\alpha_1 r_1$',r'$\alpha_2 r_2$'],loc='best')
    axes[1].plot([0,0],[-2,2],'--k',alpha=0.2)
    axes[1].plot([-2,2],[0,0],'--k',alpha=0.2)

    plt.tight_layout()


def char_solution_interactive():
    """Plots interactive characteristics solution."""
    twidget = widgets.FloatSlider(min=0.,max=2,value=0.)
    rhowidget = widgets.FloatSlider(min=0.1,max=2,value=1.,description=r'$\rho$')
    Kwidget = widgets.FloatSlider(min=0.1,max=2,value=1.)

    interact_gui = widgets.HBox([widgets.VBox([twidget]), widgets.VBox([rhowidget, Kwidget])]);

    mainwidget = interact(char_solution, t=twidget, rho=rhowidget, K=Kwidget);

    mainwidget.widget.close()
    display(interact_gui)
    display(mainwidget.widget.out)

def char_solution(t, K, rho):
    """Plotting function for char_solution_interactive."""
    fig, axes = plt.subplots(1,2,figsize=(11.5,5.5))
    c = np.sqrt(K/rho)
    x = np.linspace(-2*c-1,2*c+1,41)
    tt = np.linspace(0,2,20)
    for ix in x:
        axes[0].plot(ix-c*tt,tt,'-k',lw=0.5,color=colors[0])
        axes[0].plot(ix+c*tt,tt,'-k',lw=0.5,color=colors[1])
    axes[0].set_xlim(-1,1)
    axes[0].set_ylim(-0.2,2)
    axes[0].set_title('Characteristics')
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$t$')
    
    xx = np.linspace(-2*c-1,2*c+1,1000)
    w120 = lambda x: -0.1*np.exp(-50*x**2)
    w220 = lambda x:  0.1*np.exp(-50*x**2)
    spacing = 1
    l1, = axes[0].plot(xx,w120(xx+c*spacing*t)+spacing*t,color=colors[0],lw=2,label='$w_{1}$')
    l2, = axes[0].plot(xx,w220(xx-c*spacing*t)+spacing*t,color=colors[1],lw=2,label='$w_{2}$')
    axes[0].legend(handles=[l1,l2], loc=4)
    axes[1].plot(xx,w120(xx-c*spacing*t)+w220(xx+c*spacing*t)+spacing*t,'-k',lw=2)
    axes[1].set_xlim(-1,1)
    axes[1].set_ylim(-0.2,2)
    axes[1].set_title('Velocity')
    axes[1].set_xlabel('$x$')
    axes[1].set_ylabel('$t$')


def phase_plane_plot():
    """Plots phase plane, also used by interactive_phase_plane
    since it returns phase plane function ready to use with interact."""

    def plot_function(pl,ul,pr,ur,rho,bulk,
                      xmin=0,xmax=6,ymin=-6,ymax=6,show_phys=True,show_unphys=True):
        "Subfunction required for interactive (function of only interactive parameters)."
    
        # Define parameters
        dp = pr - pl
        du = ur - ul
        c = np.sqrt(bulk/rho)
        Z = rho*c
        
        # Define eigenvectors and functions
        eig1 = np.array([-Z, 1])
        eig2 = np.array([Z, 1])
        lin1l = lambda p: ul - 1./Z*(p-pl) 
        lin2l = lambda p: ul + 1./Z*(p-pl) 
        lin1r = lambda p: ur - 1./Z*(p-pr) 
        lin2r = lambda p: ur + 1./Z*(p-pr) 
        
        
        # Solve Riemann problem
        al1 = (-dp + du*Z)/(2*Z)
        pm = pl - al1*Z
        um = ul + al1
        
        # Set plot bounds
        fig, ax = plt.subplots(figsize=(8,5))
        x = (pl, pr, pm)
        y = (ul, ur, um)
        dx, dy = xmax - xmin, ymax - ymin
        ax.set_xlim(min(0.00000001,xmin),xmax)
        ax.set_ylim(ymin,ymax)
        ax.set_xlabel('Pressure (p)', fontsize=15)
        ax.set_ylabel('Velocity (u)', fontsize=15)

        p = np.linspace(xmin,xmax,500)

        if show_unphys:
            # Plot unphysical solutions
            ax.plot(p,lin2l(p),'--k')
            ax.plot(p,lin1r(p),'--k')

        if show_phys:
            # Plot physical solutions
            ax.plot(p,lin1l(p),'-k')
            ax.plot(p,lin2r(p),'-k')
            if (pm>=0 and pm <= xmax and um > ymin and um < ymax):
                ax.plot(pm, um, '-ok', markersize=10)
                ax.text(x[2] + 0.03*dx,y[2] + 0.03*dy, '$q_m$', fontsize=15)

        # Plot initial states and markers
        ax.plot(pl, ul, '-ok', markersize=10)
        ax.plot(pr, ur, '-ok', markersize=10)
        for i,label in enumerate(('$q_l$', '$q_r$')):
            ax.text(x[i] + 0.03*dx,y[i] + 0.03*dy,label, fontsize=15)
        plt.show()
    return plot_function


def interactive_phase_plane(ql=(10.0, -5.0), qr=(40.0, 5.0), rho=2.0, bulk=1.0):
    """Plots interactive phase plane plot."""

    # Create plot function for interact
    pp_plot = phase_plane_plot()

    # Declare all widget sliders
    ql1_widget = widgets.FloatSlider(value=ql[0],min=0.01,max=50.0, description='$p_l$')
    ql2_widget = widgets.FloatSlider(value=ql[1],min=-30,max=30.0, description='$u_l$')
    qr1_widget = widgets.FloatSlider(value=qr[0],min=0.01,max=50.0, description='$p_r$')
    qr2_widget = widgets.FloatSlider(value=qr[1],min=-30,max=30.0, description='$u_r$')
    rho_widget = widgets.FloatSlider(value=rho,min=0.01,max=10.0, description=r'$\rho$')
    bulk_widget = widgets.FloatSlider(value=bulk,min=0.01,max=10.0, description='$K$')
    xmin_widget = widgets.BoundedFloatText(value=0.0000001, description='$p_{min}:$')
    xmax_widget = widgets.FloatText(value=50, description='$p_{max}:$')
    ymin_widget = widgets.FloatText(value=-30, description='$u_{min}:$')
    ymax_widget = widgets.FloatText(value=30, description='$u_{max}:$')
    show_physical = widgets.Checkbox(value=True, description='Physical solution')
    show_unphysical = widgets.Checkbox(value=True, description='Unphysical solution')

    # Allow for dependent widgets to update
    def update_xmin(*args):
        ql1_widget.min = xmin_widget.value
        qr1_widget.min = xmin_widget.value
    def update_xmax(*args):
        ql1_widget.max = xmax_widget.value
        qr1_widget.max = xmax_widget.value
    def update_ymin(*args):
        ql2_widget.min = ymin_widget.value
        qr2_widget.min = ymin_widget.value
    def update_ymax(*args):
        ql2_widget.max = ymax_widget.value
        qr2_widget.max = ymax_widget.value
    xmin_widget.observe(update_xmin, 'value')
    xmax_widget.observe(update_xmax, 'value')
    ymin_widget.observe(update_ymin, 'value')
    ymax_widget.observe(update_ymax, 'value')

    # Organize slider widgets into boxes
    qleftright = widgets.VBox([widgets.HBox([ql1_widget, ql2_widget, rho_widget]),
                               widgets.HBox([qr1_widget, qr2_widget, bulk_widget])])
    plot_opts = widgets.VBox([widgets.HBox([show_physical, show_unphysical]),
                              widgets.HBox([xmin_widget, xmax_widget]),
                              widgets.HBox([ymin_widget, ymax_widget])])

    # Set up interactive GUI (tab style)
    interact_gui = widgets.Tab(children=[qleftright, plot_opts])
    interact_gui.set_title(0, 'Left and right states')
    interact_gui.set_title(1, 'Plot options')

    # Define interactive widget and run GUI
    ppwidget = interact(pp_plot, pl=ql1_widget, ul=ql2_widget,
                        pr=qr1_widget, ur=qr2_widget,
                        rho=rho_widget, bulk=bulk_widget,
                        xmin=xmin_widget, xmax=xmax_widget,
                        ymin=ymin_widget, ymax=ymax_widget,
                        show_phys=show_physical, show_unphys=show_unphysical)
    ppwidget.widget.close()
    display(interact_gui)
    display(ppwidget.widget.out)


def full_riemann_solution_plot():
    """Plots full Riemann solution, including the phase plane, also used by 
    full_riemann_interactive since it returns plot function ready to use with interact."""
    
    def plot_function(t,pl,ul,pr,ur,rho,bulk,which_char,
                      xmin=0,xmax=6,ymin=-6,ymax=6,show_phys=True,show_unphys=True):
        "Subfunction required for interactive (function of only interactive parameters)."
    
        # Define parameters
        dp = pr - pl
        du = ur - ul
        c = np.sqrt(bulk/rho)
        Z = rho*c
        
        # Define eigenvectors and functions
        eig1 = np.array([-Z, 1])
        eig2 = np.array([Z, 1])
        lin1l = lambda p: ul - 1./Z*(p-pl) 
        lin2l = lambda p: ul + 1./Z*(p-pl) 
        lin1r = lambda p: ur - 1./Z*(p-pr) 
        lin2r = lambda p: ur + 1./Z*(p-pr) 
        
        
        # Solve Riemann problem       
        aux = [rho,bulk]
        states, speeds, riemann_eval = acoustics.exact_riemann_solution(np.array([pl,ul]), np.array([pr,ur]), aux)
        pm = states[0][1]
        um = states[1][1]

        
        # Set figure grid
        fig = plt.figure(figsize=(11.5, 5.5))
        outer_grid = gridspec.GridSpec(1, 2, wspace=0.15, hspace=0.15)
        inner_grid = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_grid[0], wspace=0.0, hspace=0.0)
        ax1 = plt.Subplot(fig, inner_grid[0]) # x-t plane
        ax2 = plt.Subplot(fig, inner_grid[1]) # x vs pressure
        ax3 = plt.Subplot(fig, inner_grid[2]) # x vs velocity
        ax4 = plt.Subplot(fig, outer_grid[1]) # phase plane
        ax1.set_ylabel("t", fontsize=10)
        ax2.set_ylabel("pressure", fontsize=10)
        ax3.set_ylabel("velocity", fontsize=10)
        ax3.set_xlabel("x", fontsize=10)
        ax1.set_xticks([])
        ax2.set_xticks([])

        # Plot Riemann solution on ax1, ax2 and ax3
        ax = np.array([ax1, ax2, ax3])
        riemann_tools.plot_riemann(states, speeds, riemann_eval, wave_types=None, t=t, ax=ax, 
                                   layout='vertical', variable_names=['pressure', 'velocity'])
        
        # Plot characteristics on ax1 if required
        if which_char:
            plot_chars=[acoustics.lambda1, acoustics.lambda2]
            riemann_tools.plot_characteristics(riemann_eval, plot_chars[which_char-1],
                                     aux=(np.array(aux),np.array(aux)), axes=ax[0], speeds=speeds)
        
        # Plot solution in phase plane plot ion ax4
        x = (pl, pr, pm)
        y = (ul, ur, um)
        dx, dy = xmax - xmin, ymax - ymin
        ax4.set_xlim(min(0.00000001,xmin),xmax)
        ax4.set_ylim(ymin,ymax)
        ax4.set_xlabel('Pressure (p)', fontsize=10)
        ax4.set_ylabel('Velocity (u)', fontsize=10)
        ax4.set_title('Phase plane', fontsize=12)
        p = np.linspace(xmin,xmax,500)

        # Plot unphysical solutions
        if show_unphys:
            ax4.plot(p,lin2l(p),'--k')
            ax4.plot(p,lin1r(p),'--k')

        # Plot physical solutions
        if show_phys:
            ax4.plot(p,lin1l(p),'-k')
            ax4.plot(p,lin2r(p),'-k')
            if (pm>=0 and pm <= xmax and um > ymin and um < ymax):
                ax4.plot(pm, um, '-ok', markersize=10)
                ax4.text(x[2] + 0.03*dx,y[2] + 0.03*dy, '$q_m$', fontsize=15)

        # Plot initial states and markers
        ax4.plot(pl, ul, '-ok', markersize=10)
        ax4.plot(pr, ur, '-ok', markersize=10)
        for i,label in enumerate(('$q_l$', '$q_r$')):
            ax4.text(x[i] + 0.03*dx,y[i] + 0.03*dy,label, fontsize=15)
            
        # Add all plots to fig and show    
        fig.add_subplot(ax1)
        fig.add_subplot(ax2)
        fig.add_subplot(ax3)
        fig.add_subplot(ax4)
        plt.show()
    return plot_function


def full_riemann_interactive(ql=(10.0, -5.0), qr=(40.0, 5.0), rho=2.0, bulk=1.0):
    """Plots interactive full riemann solution with phase plane plot."""

    # Create plot function for interact
    pp_plot = full_riemann_solution_plot()

    # Declare all widget sliders
    t_widget = widgets.FloatSlider(value=0,min=0.0,max=1.0, description='$t$')
    ql1_widget = widgets.FloatSlider(value=ql[0],min=0.01,max=50.0, description='$p_l$')
    ql2_widget = widgets.FloatSlider(value=ql[1],min=-30,max=30.0, description='$u_l$')
    qr1_widget = widgets.FloatSlider(value=qr[0],min=0.01,max=50.0, description='$p_r$')
    qr2_widget = widgets.FloatSlider(value=qr[1],min=-30,max=30.0, description='$u_r$')
    rho_widget = widgets.FloatSlider(value=rho,min=0.01,max=10.0, description=r'$\rho$')
    bulk_widget = widgets.FloatSlider(value=bulk,min=0.01,max=10.0, description='$K$')
    xmin_widget = widgets.BoundedFloatText(value=0.0000001, description='$p_{min}:$')
    xmax_widget = widgets.FloatText(value=50, description='$p_{max}:$')
    ymin_widget = widgets.FloatText(value=-30, description='$u_{min}:$')
    ymax_widget = widgets.FloatText(value=30, description='$u_{max}:$')
    which_char_widget = widgets.Dropdown(options=[None,1,2],description='Characs.')
    show_physical = widgets.Checkbox(value=True, description='Physical solution')
    show_unphysical = widgets.Checkbox(value=True, description='Unphysical solution')

    # Allow for dependent widgets to update
    def update_xmin(*args):
        ql1_widget.min = xmin_widget.value
        qr1_widget.min = xmin_widget.value
    def update_xmax(*args):
        ql1_widget.max = xmax_widget.value
        qr1_widget.max = xmax_widget.value
    def update_ymin(*args):
        ql2_widget.min = ymin_widget.value
        qr2_widget.min = ymin_widget.value
    def update_ymax(*args):
        ql2_widget.max = ymax_widget.value
        qr2_widget.max = ymax_widget.value
    xmin_widget.observe(update_xmin, 'value')
    xmax_widget.observe(update_xmax, 'value')
    ymin_widget.observe(update_ymin, 'value')
    ymax_widget.observe(update_ymax, 'value')

    # Organize slider widgets into boxes
    qleftright = widgets.VBox([widgets.HBox([t_widget, which_char_widget]), 
                               widgets.HBox([ql1_widget, ql2_widget, rho_widget]),
                               widgets.HBox([qr1_widget, qr2_widget, bulk_widget])])
    plot_opts = widgets.VBox([widgets.HBox([show_physical, show_unphysical]),
                              widgets.HBox([xmin_widget, xmax_widget]),
                              widgets.HBox([ymin_widget, ymax_widget])])

    # Set up interactive GUI (tab style)
    interact_gui = widgets.Tab(children=[qleftright, plot_opts])
    interact_gui.set_title(0, 'Left and right states')
    interact_gui.set_title(1, 'Plot options')

    # Define interactive widget and run GUI
    ppwidget = interact(pp_plot, t=t_widget, 
                        pl=ql1_widget, ul=ql2_widget,
                        pr=qr1_widget, ur=qr2_widget,
                        rho=rho_widget, bulk=bulk_widget,
                        which_char=which_char_widget,
                        xmin=xmin_widget, xmax=xmax_widget,
                        ymin=ymin_widget, ymax=ymax_widget,
                        show_phys=show_physical, show_unphys=show_unphysical)
    ppwidget.widget.close()
    display(interact_gui)
    display(ppwidget.widget.out)
