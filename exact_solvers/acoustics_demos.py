import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import widgets, interact
from . import acoustics
colors = ['g','orange']

def char_solution_interactive():
    twidget = widgets.FloatSlider(min=0.,max=2,value=0.)
    rhowidget = widgets.FloatSlider(min=0.1,max=2,value=1.,description=r'$\rho$')
    Kwidget = widgets.FloatSlider(min=0.1,max=2,value=1.)

    interact_gui = widgets.HBox([widgets.VBox([twidget]), widgets.VBox([rhowidget, Kwidget])]);

    mainwidget = interact(char_solution, t=twidget, rho=rhowidget, K=Kwidget);

    mainwidget.widget.close()
    display(interact_gui)
    display(mainwidget.widget.out)

def char_solution(t, K, rho):
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
    l1, = axes[0].plot(xx,w120(xx+c*spacing*t)+spacing*t,color=colors[0],lw=2,label='$w_{12}$')
    l2, = axes[0].plot(xx,w220(xx-c*spacing*t)+spacing*t,color=colors[1],lw=2,label='$w_{22}$')
    axes[0].legend(handles=[l1,l2], loc=4)
    axes[1].plot(xx,w120(xx-c*spacing*t)+w220(xx+c*spacing*t)+spacing*t,'-k',lw=2)
    axes[1].set_xlim(-1,1)
    axes[1].set_ylim(-0.2,2)
    axes[1].set_title('Velocity')
    axes[1].set_xlabel('$x$')
    axes[1].set_ylabel('$t$')

def decompose_q_interactive():
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
    # Should also print the eigenvectors and the values w_1, w_2
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
    axes[1].legend(['$q$',r'$\alpha_1 r_1$',r'$\alpha_2 r_2$'],loc='best')
    axes[1].plot([0,0],[-2,2],'--k',alpha=0.2)
    axes[1].plot([-2,2],[0,0],'--k',alpha=0.2)

    plt.tight_layout()

