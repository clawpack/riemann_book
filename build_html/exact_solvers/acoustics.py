"""
Exact Riemann solvers for the acoustic wave equation in 1D.
Includes separate solvers for homogeneous and heterogeneous
media.
"""
import numpy as np

def lambda1(q, xi, aux):
    "Characteristic speed for 1-waves."
    rho, bulk = aux
    return -np.sqrt(bulk/rho)

def lambda2(q, xi, aux):
    "Characteristic speed for 2-waves."
    rho, bulk = aux
    return np.sqrt(bulk/rho)

def exact_riemann_solution(ql, qr, aux):

    # Define delta q, speeds and impedance
    rho, bulk = aux
    dq = qr - ql
    c = np.sqrt(bulk/rho)
    Z = rho*c

    # Define the 2 eigenvectors
    r1 = np.array([-Z, 1])
    r2 = np.array([Z,  1])

    alpha1 = (-dq[0] + dq[1]*Z)/(2*Z)
    alpha2 = (dq[0] + dq[1]*Z)/(2*Z)

    # Compute middle state qm
    qm = ql + alpha1*r1
    # It is equivalent to
    #qm = qr - alpha2*r2

    # Compute wave speeds
    speeds = np.zeros(2)
    speeds[0] = -c
    speeds[1] = c

    # Concatenate states for plotting
    states = np.column_stack([ql,qm,qr])

    # Calculate reval function (used for plotting the solution)
    def reval(xi):
            r"""Returns the Riemann solution for any value of xi = x/t.
            """
            p_out =  (xi<=speeds[0]                 )*ql[0]      \
                    + (xi>speeds[0])*(xi<=speeds[1])*qm[0]    \
                    + (xi>speeds[1]                 )*qr[0]

            u_out =  (xi<=speeds[0]                 )*ql[1]      \
                    + (xi>speeds[0])*(xi<=speeds[1])*qm[1]    \
                    + (xi>speeds[1]                 )*qr[1]
            return p_out, u_out

    return states, speeds, reval


# Functions for heterogeneous media

def lambda_1_het(q, xi, aux):
    "Characteristic speed for 1-waves in a heterogeneous medium."
    rho, K = aux
    return -np.sqrt(K/rho)

def lambda_2_het(q, xi, aux):
    "Characteristic speed for 2-waves in a heterogeneous medium."
    rho, K = aux
    return np.sqrt(K/rho)

def exact_riemann_heterogenous(ql, qr, auxl, auxr):

    # Define delta q, speeds and impedance (left and right)
    dq = qr - ql
    rhol, bulkl = auxl
    rhor, bulkr = auxr
    cl = np.sqrt(bulkl/rhol)
    cr = np.sqrt(bulkr/rhor)
    Zl = rhol*cl
    Zr = rhor*cr

    # Define the 2 eigenvectors
    r1 = np.array([-Zl, 1])
    r2 = np.array([Zr,  1])

    # Compute the alphas
    alpha1 = (-dq[0] + dq[1]*Zr)/(Zl + Zr)
    alpha2 = (dq[0] + dq[1]*Zl)/(Zl + Zr)

    # Compute middle state qm
    qm = ql + alpha1*r1
    # It is equivalent to
    #qm = qr - alpha2*r2

    # Compute waves speeds (characteristics)
    speeds = np.zeros(2)
    speeds[0] = -cl
    speeds[1] = cr

    # Concatenate states for plotting
    states = np.column_stack([ql,qm,qr])

    # Calculate reval function (only necessary for plotting)
    def reval(xi):
            r"""Returns the Riemann solution in for any value of xi = x/t.
            """
            p_out =  (xi<=speeds[0]                 )*ql[0]      \
                    + (xi>speeds[0])*(xi<=speeds[1])*qm[0]    \
                    + (xi>speeds[1]                 )*qr[0]

            u_out =  (xi<=speeds[0]                 )*ql[1]      \
                    + (xi>speeds[0])*(xi<=speeds[1])*qm[1]    \
                    + (xi>speeds[1]                 )*qr[1]
            return p_out, u_out

    return states, speeds, reval
