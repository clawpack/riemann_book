import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

conserved_variables = ('Depth', 'Momentum')
primitive_variables = ('Depth', 'Velocity')


def primitive_to_conservative(h, u):
    hu = h*u
    return h, hu


def conservative_to_primitive(h, hu):
    if np.any(h == 0):
        raise Exception('Attempt to divide by zero depth')
    u = hu/h
    return h, u


def exact_riemann_solution(q_l, q_r, grav=1., force_waves=None):
    """Return the exact solution to the Riemann problem with initial states q_l, q_r.
       The solution is given in terms of a list of states, a list of speeds (each of which
       may be a pair in case of a rarefaction fan), and a function reval(xi) that gives the
       solution at a point xi=x/t.

       The input and output vectors are the conserved quantities.
    """
    h_l, u_l = conservative_to_primitive(*q_l)
    h_r, u_r = conservative_to_primitive(*q_r)
    hu_l = q_l[1]
    hu_r = q_r[1]

    # Compute left and right state sound speeds
    c_l = np.sqrt(grav*h_l)
    c_r = np.sqrt(grav*h_r)

    # Define the integral curves and hugoniot loci
    integral_curve_1   = lambda h: h*u_l + \
            2*h*(np.sqrt(grav*h_l) - np.sqrt(grav*h))
    integral_curve_2   = lambda h: h*u_r - \
            2*h*(np.sqrt(grav*h_r) - np.sqrt(grav*h))
    hugoniot_locus_1 = lambda h: h_l*u_l + (h-h_l)*(u_l -
            np.sqrt(grav*h_l*(1 + (h-h_l)/h_l) * (1 + (h-h_l)/(2*h_l))))
    hugoniot_locus_2 = lambda h: h_r*u_r + (h-h_r)*(u_r +
            np.sqrt(grav*h_r*(1 + (h-h_r)/h_r) * (1 + (h-h_r)/(2*h_r))))

    # Check whether the 1-wave is a shock or rarefaction
    def phi_l(h):
        if (h>=h_l and force_waves!='raref') or force_waves=='shock':
            return hugoniot_locus_1(h)
        else:
            return integral_curve_1(h)

    # Check whether the 2-wave is a shock or rarefaction
    def phi_r(h):
        if (h>=h_r and force_waves!='raref') or force_waves=='shock':
            return hugoniot_locus_2(h)
        else:
            return integral_curve_2(h)

    phi = lambda h: phi_l(h)-phi_r(h)

    # Compute middle state h, hu by finding curve intersection
    guess = (u_l-u_r+2.*np.sqrt(grav)*(np.sqrt(h_l)+np.sqrt(h_r)))**2./16./grav
    h_m,info, ier, msg = fsolve(phi, guess, full_output=True, xtol=1.e-14)
    # For strong rarefactions, sometimes fsolve needs help
    if ier!=1:
        h_m,info, ier, msg = fsolve(phi, guess,full_output=True,factor=0.1,xtol=1.e-10)
        # This should not happen:
        if ier!=1:
            print('Warning: fsolve did not converge.')
            print(msg)

    hu_m = phi_l(h_m)
    u_m = hu_m / h_m

    # compute the wave speeds
    ws = np.zeros(4)
    wave_types = ['', '']

    # Find shock and rarefaction speeds
    if (h_m>h_l and force_waves!='raref') or force_waves=='shock':
        wave_types[0] = 'shock'
        ws[0] = (hu_l - hu_m) / (h_l - h_m)
        ws[1] = ws[0]
    else:
        wave_types[0] = 'raref'
        c_m = np.sqrt(grav * h_m)
        ws[0] = u_l - c_l
        ws[1] = u_m - c_m

    if (h_m>h_r and force_waves!='raref') or force_waves=='shock':
        wave_types[1] = 'shock'
        ws[2] = (hu_r - hu_m) / (h_r - h_m)
        ws[3] = ws[2]
    else:
        wave_types[0] = 'raref'
        c_m = np.sqrt(grav * h_m)
        ws[2] = u_m + c_m
        ws[3] = u_r + c_r

    # Find solution inside rarefaction fans (in primitive variables)
    def raref1(xi):
        RiemannInvariant = u_l + 2*np.sqrt(grav*h_l)
        h = (RiemannInvariant - xi)**2 / (9*grav)
        u = xi + np.sqrt(grav*h)
        hu = h*u
        return h, hu

    def raref2(xi):
        RiemannInvariant = u_r - 2*np.sqrt(grav*h_r)
        h = (RiemannInvariant - xi)**2 / (9*grav)
        u = xi - np.sqrt(grav*h)
        hu = h*u
        return h, hu

    q_m = np.squeeze(np.array((h_m, hu_m)))

    states = np.column_stack([q_l,q_m,q_r])
    speeds = [[], []]
    if wave_types[0] is 'shock':
        speeds[0] = ws[0]
    else:
        speeds[0] = (ws[0],ws[1])
    if wave_types[1] is 'shock':
        speeds[1] = ws[2]
    else:
        speeds[1] = (ws[2],ws[3])

    def reval(xi):
        rar1 = raref1(xi)
        rar2 = raref2(xi)
        h_out = (xi<=ws[0])*h_l + \
            (xi>ws[0])*(xi<=ws[1])*rar1[0] + \
            (xi>ws[1])*(xi<=ws[2])*h_m +  \
            (xi>ws[2])*(xi<=ws[3])*rar2[0] +  \
            (xi>ws[3])*h_r
        hu_out = (xi<=ws[0])*hu_l + \
            (xi>ws[0])*(xi<=ws[1])*rar1[1] + \
            (xi>ws[1])*(xi<=ws[2])*hu_m +  \
            (xi>ws[2])*(xi<=ws[3])*rar2[1] +  \
            (xi>ws[3])*hu_r
        return h_out, hu_out

    return states, speeds, reval, wave_types


def integral_curve(h, hstar, hustar, wave_family, g=1.):
    """
    Return hu as a function of h for integral curves through
    (hstar, hustar).
    """
    ustar = hustar / hstar
    if wave_family == 1:
        hu = h*ustar + 2*h*(np.sqrt(g*hstar) - np.sqrt(g*h))
    else:
        hu = h*ustar - 2*h*(np.sqrt(g*hstar) - np.sqrt(g*h))
    return hu


def hugoniot_locus(h, hstar, hustar, wave_family, g=1.):
    """
    Return hu as a function of h for the Hugoniot locus through
    (hstar, hustar).
    """
    ustar = hustar / hstar
    alpha = h - hstar
    d = np.sqrt(g*hstar*(1 + alpha/hstar)*(1 + alpha/(2*hstar)))
    if wave_family == 1:
        hu = hustar + alpha*(ustar - d)
    else:
        hu = hustar + alpha*(ustar + d)
    return hu


def phase_plane_curves(hstar, hustar, state, wave_family='both'):
    """
    Plot the curves of points in the h - hu phase plane that can be connected to (hstar,hustar).
    state = 'qleft' or 'qright' indicates whether the specified state is ql or qr.
    wave_family = 1, 2, or 'both' indicates whether 1-waves or 2-waves should be plotted.
    Colors in the plots indicate whether the states can be connected via a shock or rarefaction.
    """

    h = np.linspace(0, hstar, 200)

    if wave_family in [1,'both']:
        if state == 'qleft':
            hu = integral_curve(h, hstar, hustar, 1)
            plt.plot(h,hu,'b', label='1-rarefactions')
        else:
            hu = hugoniot_locus(h, hstar, hustar, 1)
            plt.plot(h,hu,'r', label='1-shocks')

    if wave_family in [2,'both']:
        if state == 'qleft':
            hu = hugoniot_locus(h, hstar, hustar, 2)
            plt.plot(h,hu,'g', label='2-shocks')
        else:
            hu = integral_curve(h, hstar, hustar, 2)
            plt.plot(h,hu,'m', label='2-rarefactions')

    h = np.linspace(hstar, 5, 200)

    if wave_family in [1,'both']:
        if state == 'qright':
            hu = integral_curve(h, hstar, hustar, 1)
            plt.plot(h,hu,'b', label='1-rarefactions')
        else:
            hu = hugoniot_locus(h, hstar, hustar, 1)
            plt.plot(h,hu,'r', label='1-shocks')

    if wave_family in [2,'both']:
        if state == 'qright':
            hu = hugoniot_locus(h, hstar, hustar, 2)
            plt.plot(h,hu,'g', label='2-shocks')
        else:
            hu = integral_curve(h, hstar, hustar, 2)
            plt.plot(h,hu,'m', label='2-rarefactions')

    # plot and label the point (hstar, hustar)
    plt.plot([hstar],[hustar],'ko',markersize=5)
    plt.text(hstar + 0.1, hustar - 0.2, state, fontsize=13)


def make_axes_and_label(x1=-.5, x2=6., y1=-2.5, y2=2.5):
    plt.plot([x1,x2],[0,0],'k')
    plt.plot([0,0],[y1,y2],'k')
    plt.axis([x1,x2,y1,y2])
    plt.legend()
    plt.xlabel("h = depth",fontsize=15)
    plt.ylabel("hu = momentum",fontsize=15)

def phase_plane_plot(q_l, q_r, g=1., ax=None, force_waves=None, y_axis='u'):
    r"""Plot the Hugoniot loci or integral curves in the h-u or h-hu plane."""
    # Solve Riemann problem
    states, speeds, reval, wave_types = \
                        exact_riemann_solution(q_l, q_r, g, force_waves=force_waves)

    # Set plot bounds
    if ax is None:
        fig, ax = plt.subplots()
    x = states[0,:]
    if y_axis == 'hu':
        y = states[1,:]
    else:
        y = states[1,:]/states[0,:]

    xmax, xmin = max(x), min(x)
    ymax, ymin = max(y), min(y)
    dx, dy = xmax - xmin, ymax - ymin
    ymax = max(abs(y))
    #ax.set_xlim(xmin - 0.1*dx, xmax + 0.1*dx)
    ax.set_xlim(0, xmax + 0.5*dx)
    #ax.set_ylim(ymin - 0.1*dy, ymax + 0.1*dy)
    ax.set_ylim(-ymax - 0.5*dy, ymax + 0.5*dy)
    ax.set_xlabel('Depth (h)')
    if y_axis == 'u':
        ax.set_ylabel('Velocity (u)')
    else:
        ax.set_ylabel('Momentum (hu)')

    left, middle, right = (0, 1, 2)
    # Plot curves
    #h = np.linspace(min(states[0,left],states[0,middle]),max(states[0,left],states[0,middle]))
    h_l = states[0,left]
    h1 = np.linspace(1.e-2,h_l)
    h2 = np.linspace(h_l,xmax+0.5*dx)
    if wave_types[0] == 'shock':
        hu1 = hugoniot_locus(h1, states[0,left], states[1,left], wave_family=1, g=g)
        hu2 = hugoniot_locus(h2, states[0,left], states[1,left], wave_family=1, g=g)
        if y_axis == 'u':
            hu1 = hu1/h1
            hu2 = hu2/h2
        ax.plot(h1,hu1,'--r')
        ax.plot(h2,hu2,'r')
    else:
        hu1 = integral_curve(h1, states[0,left], states[1,left], wave_family=1, g=g)
        hu2 = integral_curve(h2, states[0,left], states[1,left], wave_family=1, g=g)
        if y_axis == 'u':
            hu1 = hu1/h1
            hu2 = hu2/h2
        ax.plot(h1,hu1,'b')
        ax.plot(h2,hu2,'--b')

    h_r = states[0,right]
    h1 = np.linspace(1.e-2,h_r)
    h2 = np.linspace(h_r,xmax+0.5*dx)
    if wave_types[1] == 'shock':
        hu1 = hugoniot_locus(h1, states[0,right], states[1,right], wave_family=2, g=g)
        hu2 = hugoniot_locus(h2, states[0,right], states[1,right], wave_family=2, g=g)
        if y_axis == 'u':
            hu1 = hu1/h1
            hu2 = hu2/h2
        ax.plot(h1,hu1,'--r')
        ax.plot(h2,hu2,'r')
    else:
        hu1 = integral_curve(h1, states[0,right], states[1,right], wave_family=2, g=g)
        hu2 = integral_curve(h2, states[0,right], states[1,right], wave_family=2, g=g)
        if y_axis == 'u':
            hu1 = hu1/h1
            hu2 = hu2/h2
        ax.plot(h1,hu1,'b')
        ax.plot(h2,hu2,'--b')

    for xp,yp in zip(x,y):
        ax.plot(xp,yp,'ok',markersize=10)
    # Label states
    for i,label in enumerate(('Left', 'Middle', 'Right')):
        ax.text(x[i] + 0.025*dx,y[i] + 0.025*dy,label)

def plot_hugoniot_loci(plot_1=True,plot_2=False,y_axis='hu'):
    h = np.linspace(0.001,3,100)
    hstar = 1.0
    legend = plot_1*['1-loci'] + plot_2*['2-loci']
    for hustar in np.linspace(-4,4,15):
        if plot_1:
            hu = hugoniot_locus(h,hstar,hustar,wave_family=1)
            if y_axis=='hu':
                plt.plot(h,hu,'-',color='coral')
            else:
                u = hu/h
                plt.plot(h,u,'-',color='coral')
        if plot_2:
            hu = hugoniot_locus(h,hstar,hustar,wave_family=2)
            if y_axis=='hu':
                plt.plot(h,hu,'-',color='maroon')
            else:
                u = hu/h
                plt.plot(h,u,'-',color='maroon')
        plt.axis((0,3,-3,3))
        plt.xlabel('depth h')
        if y_axis=='hu':
            plt.ylabel('momentum hu')
        else:
            plt.ylabel('velocity u')
        plt.title('Hugoniot loci')
        plt.legend(legend,loc=1)
    plt.show()
