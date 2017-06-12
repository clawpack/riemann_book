
import matplotlib.pyplot as plt
import numpy as np

def figure1():
    x = np.linspace(-1,1,21)
    t = np.linspace(0,1,20)
    a = 1.
    for ix in x:
        plt.plot(ix+a*t,t,'-k',lw=0.5)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title('Characteristics')
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    xx = np.linspace(-1,1,1000)

def figure2():
    figure1()
    plt.title('Propagation along characteristics')
    xx = np.linspace(-1,1,1000)
    q = 0.1*np.exp(-100*(xx-0.2)**2)
    plt.plot(xx,q,'-r')
    spacing = 0.04
    number = 20
    for i in range(number):
        plt.plot(xx+spacing*i,q+spacing*i,'-r')

def figure3():
    figure1()
    plt.xlim(-0.2,0.8)
    plt.title('Solution of the Riemann problem')
    xx = np.linspace(-2,1,1000)
    #q = 0.05+0.05*(xx<0.)
    q = 0.05*(xx<0.)
    spacing = 0.04
    number = 20
    for i in range(number):
        plt.plot(xx+spacing*i,q+spacing*i,'-r')
