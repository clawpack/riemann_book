#
# Demonstrates shock wave and rarefaction cases for Burgers' equation
# using Riemann's method. The output for both cases are written as
# simple snapshots for selected time values which can then be combined
# into an animation using ImageMagick.
# convert -delay 20 -loop 0 /tmp/shock*.png /tmp/shock.gif
# convert -delay 20 -loop 0 /tmp/rarefaction*.png /tmp/rarefaction.gif
#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def speed(q, xi):
    return q

def exact_riemann_solution(xi,q_l,q_r):
    f = lambda q: 0.5*q*q
    # Shock wave
    if q_l > q_r: 
        shock_speed = (f(q_l)-f(q_r))/(q_l-q_r)
        q = (xi < shock_speed)*q_l \
          + (xi >=shock_speed)*q_r
        return q
    # Rarefaction wave
    else:  
        c_l  = q_l
        c_r = q_r
        q = (xi<=c_l)*q_l \
          + (xi>=c_r)*q_r \
          + (c_l<xi)*(xi<c_r)*xi
        return q
    
def rarefaction():
    print ('rarefaction')
    q_l, q_r = 2.0, 4.0
    
    for t in np.linspace(0,1,6):
        outfile = '/tmp/rarefaction-%f.png' % t
        print (outfile)

        fig, ax = plt.subplots(figsize=(5, 3))
                    
        x = np.linspace(-4, 4, 1000)
        
        q = np.array([exact_riemann_solution(xi/(t+1e-10),q_l,q_r) for xi in x])

        ax.set_xlim(-4,4)

        ax.plot(x,q,'-k',lw=2)
    
        plt.savefig(outfile) 
        
        #plot_riemann(reval, t, outfile)

def shock():
    print ('shock wave')
    q_l, q_r = 5.0, 1.0

    for t in np.linspace(0,1,6):
        outfile = '/tmp/shock-%f.png' % t
        print (outfile)

        fig, ax = plt.subplots(figsize=(5, 3))
                    
        x = np.linspace(-4, 4, 1000)
        
        q = np.array([exact_riemann_solution(xi/(t+1e-10),q_l,q_r) for xi in x])

        ax.set_xlim(-4,4)

        ax.plot(x,q,'-k',lw=2)
    
        plt.savefig(outfile) 

        
if __name__ == "__main__": 
    
    rarefaction()
    shock()


