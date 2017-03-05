import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

conserved_variables = ('Density', 'Momentum', 'Energy')
primitive_variables = ('Density', 'Velocity', 'Pressure')


def primitive_to_conservative(rho, u, p, gamma=1.4):
    mom = rho*u
    E   = p/(gamma-1.) + 0.5*rho*u**2
    return rho, mom, E


def conservative_to_primitive(rho, mom, E, gamma=1.4):
    u = mom/rho
    p = (gamma-1.)*(E - 0.5*rho*u**2)
    return rho, u, p


def exact_riemann_solution(q_l, q_r, gamma=1.4, phase_plane_curves=False):
    """Return the exact solution to the Riemann problem with initial states
       q_l, q_r.  The solution is given in terms of a list of states, a list of
       speeds (each of which may be a pair in case of a rarefaction fan), and a
       function reval(xi) that gives the solution at a point xi=x/t.

       The input and output vectors are the conserved quantities.

       If phase_plane_curves==True, then the appropriate Hugoniot Locus and/or
       integral curve is returned for the 1- and 3-waves.
    """

    rho_l, u_l, p_l = conservative_to_primitive(*q_l)
    rho_r, u_r, p_r = conservative_to_primitive(*q_r)

    # Compute left and right state sound speeds
    c_l = np.sqrt(gamma*p_l/rho_l)
    c_r = np.sqrt(gamma*p_r/rho_r)

    beta = (gamma+1.)/(gamma-1.)

    # Check for cavitation
    if u_l - u_r + 2*(c_l+c_r)/(gamma-1.) < 0:
        print('Cavitation detected!  Exiting.')
        return None

    # Define the integral curves and hugoniot loci
    integral_curve_1   = lambda p : u_l + 2*c_l/(gamma-1.)*(1.-(max(p,0)/p_l)**((gamma-1.)/(2.*gamma)))
    integral_curve_3   = lambda p : u_r - 2*c_r/(gamma-1.)*(1.-(max(p,0)/p_r)**((gamma-1.)/(2.*gamma)))
    hugoniot_locus_1 = lambda p : u_l + 2*c_l/np.sqrt(2*gamma*(gamma-1.)) * ((1-p/p_l)/np.sqrt(1+beta*p/p_l))
    hugoniot_locus_3 = lambda p : u_r - 2*c_r/np.sqrt(2*gamma*(gamma-1.)) * ((1-p/p_r)/np.sqrt(1+beta*p/p_r))

    # Check whether the 1-wave is a shock or rarefaction
    def phi_l(p):
        if p >= p_l: return hugoniot_locus_1(p)
        else: return integral_curve_1(p)

    # Check whether the 1-wave is a shock or rarefaction
    def phi_r(p):
        if p >= p_r: return hugoniot_locus_3(p)
        else: return integral_curve_3(p)

    phi = lambda p : phi_l(p)-phi_r(p)

    # Compute middle state p, u by finding curve intersection
    p, info, ier, msg = fsolve(phi, (p_l+p_r)/2., full_output=True, xtol=1.e-14)
    # For strong rarefactions, sometimes fsolve needs help
    if ier != 1:
        p, info, ier, msg = fsolve(phi, (p_l+p_r)/2., full_output=True, factor=0.1, xtol=1.e-10)
        # This should not happen:
        if ier != 1:
            print('Warning: fsolve did not converge.')
            print(msg)

    u = phi_l(p)

    # Find middle state densities
    rho_l_star = (p/p_l)**(1./gamma) * rho_l
    rho_r_star = (p/p_r)**(1./gamma) * rho_r

    # compute the wave speeds
    ws = np.zeros(5)
    # The contact speed:
    ws[2] = u

    # Find shock and rarefaction speeds
    if p > p_l:
        ws[0] = (rho_l*u_l - rho_l_star*u)/(rho_l - rho_l_star)
        ws[1] = ws[0]
    else:
        c_l_star = np.sqrt(gamma*p/rho_l_star)
        ws[0] = u_l - c_l
        ws[1] = u - c_l_star

    if p > p_r:
        ws[4] = (rho_r*u_r - rho_r_star*u)/(rho_r - rho_r_star)
        ws[3] = ws[4]
    else:
        c_r_star = np.sqrt(gamma*p/rho_r_star)
        ws[3] = u+c_r_star
        ws[4] = u_r + c_r

    # Find solution inside rarefaction fans (in primitive variables)
    def raref1(xi):
        u1 = ((gamma-1.)*u_l + 2*(c_l + xi))/(gamma+1.)
        rho1 = (rho_l**gamma*(u1-xi)**2/(gamma*p_l))**(1./(gamma-1.))
        p1 = p_l*(rho1/rho_l)**gamma
        return rho1, u1, p1

    def raref3(xi):
        u3 = ((gamma-1.)*u_r - 2*(c_r - xi))/(gamma+1.)
        rho3 = (rho_r**gamma*(xi-u3)**2/(gamma*p_r))**(1./(gamma-1.))
        p3 = p_r*(rho3/rho_r)**gamma
        return rho3, u3, p3

    q_l_star = np.squeeze(np.array(primitive_to_conservative(rho_l_star,u,p)))
    q_r_star = np.squeeze(np.array(primitive_to_conservative(rho_r_star,u,p)))

    states = np.column_stack([q_l,q_l_star,q_r_star,q_r])
    speeds = [(ws[0],ws[1]),ws[2],(ws[3],ws[4])]

    def reval(xi):
        r"""Returns the Riemann solution in primitive variables for any
            value of xi = x/t.
        """
        rar1 = raref1(xi)
        rar3 = raref3(xi)
        rho_out =  (xi<=speeds[0][0]                  )*rho_l      \
                 + (xi>speeds[0][0])*(xi<=speeds[0][1])*rar1[0]    \
                 + (xi>speeds[0][1])*(xi<=speeds[1]   )*rho_l_star \
                 + (xi>speeds[1])   *(xi<=speeds[2][0])*rho_r_star \
                 + (xi>speeds[2][0])*(xi<=speeds[2][1])*rar3[0]    \
                 + (xi>speeds[2][1]                   )*rho_r

        u_out   =  (xi<=speeds[0][0]                  )*u_l     \
                 + (xi>speeds[0][0])*(xi<=speeds[0][1])*rar1[1] \
                 + (xi>speeds[0][1])*(xi<=speeds[1]   )*u       \
                 + (xi>speeds[1]   )*(xi<=speeds[2][0])*u       \
                 + (xi>speeds[2][0])*(xi<=speeds[2][1])*rar3[1] \
                 + (xi>speeds[2][1]                   )*u_r

        p_out   =  (xi<=speeds[0][0]                  )*p_l     \
                 + (xi>speeds[0][0])*(xi<=speeds[0][1])*rar1[2] \
                 + (xi>speeds[0][1])*(xi<=speeds[1]   )*p       \
                 + (xi>speeds[1]   )*(xi<=speeds[2][0])*p       \
                 + (xi>speeds[2][0])*(xi<=speeds[2][1])*rar3[2] \
                 + (xi>speeds[2][1]                   )*p_r
        return primitive_to_conservative(rho_out,u_out,p_out)

    if phase_plane_curves:
        return states, speeds, reval, (p, phi_l, phi_r)
    else:
        return states, speeds, reval


def phase_plane_plot(left_state, right_state, gamma=1.4):
    r"""Plot the Hugoniot loci or integral curves in the p-u plane."""
    # Solve Riemann problem
    q_left  = primitive_to_conservative(*left_state)
    q_right = primitive_to_conservative(*right_state)
    ex_states, ex_speeds, reval, ppc = exact_riemann_solution(q_left,
                                                              q_right,
                                                              gamma,
                                                              phase_plane_curves=True)
    pm, w1, w3 = ppc

    # Set plot bounds
    fig, ax = plt.subplots()
    x = (left_state.Pressure, pm, right_state.Pressure)
    y = (left_state.Velocity, w1(pm), right_state.Velocity)
    xmax, xmin = max(x), min(x)
    ymax, ymin = max(y), min(y)
    dx, dy = xmax - xmin, ymax - ymin
    ax.set_xlim(xmin - 0.1*dx, xmax + 0.1*dx)
    ax.set_ylim(ymin - 0.1*dy, ymax + 0.1*dy)
    ax.set_xlabel('Pressure (p)')
    ax.set_ylabel('Velocity (u)')

    # Plot curves
    w1v, w3v = (np.vectorize(w1), np.vectorize(w3))
    press1 = np.linspace(left_state.Pressure, pm)
    press3 = np.linspace(right_state.Pressure, pm)
    ax.plot(press1,w1v(press1),'k',lw=2)
    ax.plot(press3,w3v(press3),'k',lw=2)
    for xp,yp in zip(x,y):
        ax.plot(xp,yp,'or',markersize=10)
    # Label states
    for i,label in enumerate(('Left', 'Middle', 'Right')):
        ax.text(x[i] + 0.025*dx,y[i] + 0.025*dy,label)
