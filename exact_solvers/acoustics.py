import numpy as np
import matplotlib.pyplot as plt

def lambda1(q, xi, aux):
    "Characteristic speed for 1-waves."
    bulk = aux[0]
    rho = aux[1]
    return -np.sqrt(K/rho)

def lambda2(q, xi, aux):
    "Characteristic speed for 2-waves."
    bulk = aux[0]
    rho = aux[1]
    return np.sqrt(K/rho)

def exact_riemann_solution(ql,qr,bulk,rho):

    # Define delat q, speeds and impedance
    dq = qr - ql
    c = np.sqrt(bulk/rho)
    Z = rho*c

    # Define the 2 eigenvectors
    r1 = np.array([-Z, 1])
    r2 = np.array([Z,  1])

    # Compute the alphas
    alpha1 = (-dq[0] + dq[1]*Z)/(2*Z)
    alpha2 = (dq[0] + dq[1]*Z)/(2*Z)

    # Compute middle state qm
    qm = ql + alpha1*r1 
    # It is equivalent to
    #qm = qr - alpha2*r2 
    
    # Compute waves speeds (characteristics) 
    wspeeds = np.zeros(2)
    wspeeds[0] = -c
    wspeeds[1] = c
    
    # Concatenate states for plotting
    states = np.column_stack([ql,qm,qr])

    # Calculate reval function (only necessary for plotting)
    def reval(xi):
            r"""Returns the Riemann solution in for any value of xi = x/t.
            """
            p_out =  (xi<=wspeeds[0]                 )*ql[0]      \
                    + (xi>wspeeds[0])*(xi<=wspeeds[1])*qm[0]    \
                    + (xi>wspeeds[1]                 )*qr[0] 

            u_out =  (xi<=wspeeds[0]                 )*ql[1]      \
                    + (xi>wspeeds[0])*(xi<=wspeeds[1])*qm[1]    \
                    + (xi>wspeeds[1]                 )*qr[1] 
            return p_out, u_out
       
    return states, wspeeds, reval

