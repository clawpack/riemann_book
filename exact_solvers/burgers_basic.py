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

def exact_riemann_solution(q_l,q_r):
    f = lambda q: 0.5*q*q
    states = np.array([[q_l, q_r]])
    # Shock wave
    if q_l > q_r: 
        shock_speed = (f(q_l)-f(q_r))/(q_l-q_r)
        speeds = [[shock_speed,shock_speed]]
        def reval(xi):
            q = np.zeros((1,len(xi)))
            q[0,:] = (xi < shock_speed)*q_l \
              + (xi >=shock_speed)*q_r
            return q
    # Rarefaction wave
    else:  
        c_l  = q_l
        c_r = q_r

        speeds = [[c_l,c_r]]

        def reval(xi):
            q = np.zeros((1,len(xi)))
            q[0,:] = (xi<=c_l)*q_l \
              + (xi>=c_r)*q_r \
              + (c_l<xi)*(xi<c_r)*xi
            return q

    return states, speeds, reval
    
def rarefaction():
    print ('rarefaction')
    q_l, q_r = 2.0, 4.0
    states, speeds, reval = exact_riemann_solution(q_l ,q_r)
    for t in np.linspace(0,1,6):
        outfile = '/tmp/rarefaction-%f.png' % t
        print (outfile)
        plot_riemann(states, speeds, reval, t, outfile)

def shock():
    print ('shock wave')
    q_l, q_r = 5.0, 1.0
    states, speeds, reval = exact_riemann_solution(q_l ,q_r)
    for t in np.linspace(0,1,6):
        outfile = '/tmp/shock-%f.png' % t
        print (outfile)
        plot_riemann(states, speeds, reval, t, outfile)

    
def plot_riemann(states, s, riemann_eval, t, outfile):
    
    fig, ax = plt.subplots(figsize=(5, 3))

    num_vars, num_states = states.shape
    
    tmax = 1.0
    xmax = 0.
    
    speeds = np.linspace(s[0][0],s[0][1],5)
    for ss in speeds:
        x1 = tmax * ss
        xmax = max(xmax,abs(x1))

    xmax = max(0.001, xmax)
    ax.set_xlim(-xmax,xmax)

    xi_range = np.linspace(min(-10, 2*np.min(s[0])), max(10, 2*np.max(s[-1])))
    q_sample = riemann_eval(xi_range)
            
    x = np.linspace(-xmax, xmax, 1000)

    wavespeeds = np.array(s[0])
        
    xm = 0.5 * (wavespeeds[1:]+wavespeeds[:-1]) * t
    
    iloc = np.searchsorted(x,xm)

    x = np.insert(x, iloc, xm)

    q = riemann_eval(x/(t+1e-10))

    ax.plot(x,q[0][:],'-k',lw=2)
    
    plt.savefig(outfile) 

if __name__ == "__main__": 
    
    rarefaction()
    shock()


