from __future__ import print_function
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from utils import riemann_tools
from ipywidgets import interact, interactive, widgets
from IPython.display import display

conserved_variables = ('Density', 'Momentum', 'Energy')
primitive_variables = ('Density', 'Velocity', 'Pressure')


def primitive_to_conservative(rho, u, p, gamma, pinf):
    mom = rho*u
    E   = (p - gamma * pinf)/(gamma-1.) + 0.5*rho*u**2
    return rho, mom, E


def conservative_to_primitive(rho, mom, E, gamma, pinf):
    u = mom/rho
    p = (gamma-1.)*(E - 0.5*rho*u**2) - gamma*pinf
    return rho, u, p


def exact_riemann_solution(ql, qr, gamma, pinf, varin = 'primitive', varout = 'primitive'):

    # Get intial data
    gammal, gammar = gamma
    pinfl, pinfr = pinf
    if varin == 'conservative':
        rhol, ul, pl = conservative_to_primitive(*ql, gamma = gammal, pinf = pinfl)
        rhor, ur, pr = conservative_to_primitive(*qr, gamma = gammar, pinf = pinfr)
    else:
        rhol, ul, pl = ql
        rhor, ur, pr = qr

    # Bar pressure (convenient change of variable)
    pbl = pl + pinfl
    pbr = pr + pinfr

    # Useful parameters
    gl1 = gammal - 1.0
    gr1 = gammar - 1.0
    bl = (gammal + 1.0)/(gammal - 1.0)
    br = (gammar + 1.0)/(gammar - 1.0)
    betal = pbl/bl
    betar = pbr/br
    al = 2.0/((gammal + 1.0)*rhol)
    ar = 2.0/((gammar + 1.0)*rhor)
    # Calculate velocities (sound speed)
    cl =  np.sqrt(gammal*(pl + pinfl)/rhol)
    cr =  np.sqrt(gammar*(pr + pinfr)/rhor)

    # Functions to calculate integral curves (rarefactions) and hugoniot locii (shocks)
    integral_curve_1 = lambda p : ul + 2*cl/gl1*(1 - ((p + pinfl)/pbl)**(gl1/(2.0*gammal)))
    integral_curve_3 = lambda p : ur - 2*cr/gr1*(1 - ((p + pinfr)/pbr)**(gr1/(2.0*gammar)))
    hugoniot_locus_1 = lambda p : ul - (p - pl)*np.sqrt(al/(p + pinfl + betal))
    hugoniot_locus_3 = lambda p : ur + (p - pr)*np.sqrt(ar/(p + pinfr + betar))

    # Check whether the 1-wave is a shock or rarefaction
    def phi_l(p):
        global wave1
        if p >= pl:  # 1-Shock
            wave1 = 'shock'
            return hugoniot_locus_1(p)
        else:  # 1-Rarefaction
            wave1 = 'raref'
            return integral_curve_1(p)

    # Check whether the 3-wave is a shock or rarefaction
    def phi_r(p):
        global wave3
        if p >= pr:
            wave3 = 'shock'
            return hugoniot_locus_3(p)
        else:
            wave3 = 'raref'
            return integral_curve_3(p)

    phi = lambda p : phi_l(p) - phi_r(p)

    # Use fsolve to find p_star such that Phi(p_star)=0
    p0 = (pl + pr)/2.0  # initial guess is the average of initial pressures
    p_star, info, ier, msg = fsolve(phi, p0, full_output=True, xtol=1.e-14)
    # For strong rarefactions, sometimes fsolve needs help
    if ier != 1:
        p_star, info, ier, msg = fsolve(phi, p0, full_output=True, factor=0.1, xtol=1.e-10)
        # This should not happen:
        if ier != 1:
            print('Warning: fsolve did not converge.')
            print(msg)

    # Calculate middle states ustar and rho_star in terms of p_star
    pbsl = p_star + pinfl
    pbsr = p_star + pinfr
    u_star = 0.5*(phi_l(p_star) + phi_r(p_star))
    if wave1 == 'shock':
        rhol_star = rhol*(pbsl/pbl + 1.0/bl)/(pbsl/(pbl*bl) + 1.0)
    elif wave1 == 'raref':
        rhol_star = rhol*(pbsl/pbl)**(1.0/gammal)
    if wave3 == 'shock':
        rhor_star = rhor*(pbsr/pbr + 1.0/br)/(pbsr/(pbr*br) + 1.0)
    elif wave3 == 'raref':
        rhor_star = rhor*(pbsr/pbr)**(1.0/gammar)

    # Arrange final states for output
    # Output correct name of variables
    if varout == 'conservative':
        outvars = conserved_variables
        ql      = primitive_to_conservative(rhol, ul, pl, gammal, pinfl)
        ql_star = primitive_to_conservative(rhol_star, u_star, p_star, gammal, pinfl)
        qr_star = primitive_to_conservative(rhor_star, u_star, p_star, gammar, pinfr)
        qr      = primitive_to_conservative(rhor, ur, pr, gammar, pinfr)
    else:
        outvars = primitive_variables
        ql      = [rhol, ul, pl]
        ql_star = [rhol_star, u_star, p_star]
        qr_star = [rhor_star, u_star, p_star]
        qr      = [rhor, ur, pr]
    states = np.column_stack([ql,ql_star,qr_star,qr])

    # Calculate wave speeds for output and rho_star states
    ws = np.zeros(5)
    betal = (pl + pinfl)*(gammal - 1.0)/(gammal + 1.0)
    betar = (pr + pinfr)*(gammar - 1.0)/(gammar + 1.0)
    alphal = 2.0/(rhol*(gammal + 1.0))
    alphar = 2.0/(rhor*(gammar + 1.0))
    cl_star = np.sqrt(gammal*(pbsl)/rhol_star)
    cr_star = np.sqrt(gammar*(pbsr)/rhor_star)
    ws[2] = u_star  # Contact discontinuity
    if wave1 == 'shock':
        ws[0] = ul - np.sqrt((pbsl + betal)/alphal)/rhol
        ws[1] = ws[0]
    elif wave1 == 'raref':
        ws[0] = ul - cl
        ws[1] = u_star - cl_star
    if wave3 == 'shock':
        ws[3] = ur + np.sqrt((pbsr + betar)/alphar)/rhor
        ws[4] = ws[3]
    elif wave3 == 'raref':
        ws[3] = u_star + cr_star
        ws[4] = ur + cr

    #speeds = [(ws[0],ws[1]),ws[2],(ws[3],ws[4])]
    speeds = [[], ws[2], []]
    wave_types = [wave1, 'contact', wave3]
    if wave_types[0] is 'shock':
        speeds[0] = ws[0]
    else:
        speeds[0] = (ws[0],ws[1])
    if wave_types[2] is 'shock':
        speeds[2] = ws[3]
    else:
        speeds[2] = (ws[3],ws[4])

    # Functions to find solution inside rarefaction fans
    def raref1(xi):
        u1 = (ul*gl1 + 2*(xi +  cl))/(gammal + 1.)
        rho1 = rhol*(abs(u1 - xi)/cl)**(2.0/gl1)
        p1 = pbl*(abs(u1 - xi)/cl)**(2.0*gammal/gl1) - pinfl
        return rho1, u1, p1

    def raref3(xi):
        u3 = (ur*gr1 + 2*(xi -  cr))/(gammar + 1.)
        rho3 = rhor*(abs(xi - u3)/cr)**(2.0/gr1)
        p3 = pbr*(abs(xi - u3)/cr)**(2.0*gammar/gr1) - pinfr
        return rho3, u3, p3

    #Returns the Riemann solution in primitive variables for any value of xi = x/t.
    def reval(xi):
        rar1 = raref1(xi)
        rar3 = raref3(xi)
        rho_out =  (xi<=ws[0]                  )*rhol       \
                 + (xi>ws[0])*(xi<=ws[1])*rar1[0]    \
                 + (xi>ws[1])*(xi<=speeds[1]   )*rhol_star  \
                 + (xi>speeds[1]   )*(xi<=ws[3])*rhor_star  \
                 + (xi>ws[3])*(xi<=ws[4])*rar3[0]    \
                 + (xi>ws[4]                   )*rhor

        u_out   =  (xi<=ws[0]                  )*ul      \
                 + (xi>ws[0])*(xi<=ws[1])*rar1[1] \
                 + (xi>ws[1])*(xi<=speeds[1]   )*u_star  \
                 + (xi>speeds[1]   )*(xi<=ws[3])*u_star  \
                 + (xi>ws[3])*(xi<=ws[4])*rar3[1] \
                 + (xi>ws[4]                   )*ur

        p_out   =  (xi<=ws[0]                  )*pl      \
                 + (xi>ws[0])*(xi<=ws[1])*rar1[2] \
                 + (xi>ws[1])*(xi<=speeds[1]   )*p_star  \
                 + (xi>speeds[1]   )*(xi<=ws[3])*p_star  \
                 + (xi>ws[3])*(xi<=ws[4])*rar3[2] \
                 + (xi>ws[4]                   )*pr
        gamma   =  (xi<=0                             )*gammal  \
                 + (xi>0                              )*gammar
        pinf    =  (xi<=0                             )*pinfl  \
                 + (xi>0                              )*pinfr 
        if varout == 'conservative':
            return primitive_to_conservative(rho_out,u_out,p_out,gamma,pinf)
        else:
            return rho_out,u_out,p_out

    return states, speeds, reval, wave_types, outvars


# Define hugiont locus and intergal curves independently (needed for interact version)
def hugoniot_locus_1(p,ql,params):
    gammal, pinfl = params
    rhol, ul, pl = ql
    betal = (pl + pinfl)*(gammal - 1.0)/(gammal + 1.0)
    alphal = 2.0/((gammal + 1.0)*rhol)
    return ul - (p - pl)*np.sqrt(alphal/(p + pinfl + betal))

def hugoniot_locus_3(p,qr,params):
    gammar, pinfr =  params
    rhor, ur, pr = qr
    betar = (pr + pinfr)*(gammar - 1.0)/(gammar + 1.0)
    alphar = 2.0/((gammar + 1.0)*rhor)
    return ur + (p - pr)*np.sqrt(alphar/(p + pinfr + betar))

def integral_curve_1(p,ql,params):
    gammal, pinfl =  params
    rhol, ul, pl = ql
    cl =  np.sqrt(gammal*(pl + pinfl)/rhol)
    gl1 = gammal - 1.0
    return  ul + 2*cl/gl1*(1 - ((p + pinfl)/(pl+pinfl))**(gl1/(2.0*gammal)))

def integral_curve_3(p,qr,params):
    gammar, pinfr =  params
    rhor, ur, pr = qr
    cr =  np.sqrt(gammar*(pr + pinfr)/rhor)
    gr1 = gammar - 1.0
    return  ur - 2*cr/gr1*(1 - ((p + pinfr)/(pr + pinfr))**(gr1/(2.0*gammar)))



# Function to return plotting function for interact
def phase_plane_plot(hugoniot_loci=None, integral_curves=None):
    
    # Subfunction required for interactive (only interactive parameters)
    def plot_function(rhol,ul,pl,rhor,ur,pr,gammal,pinfl,gammar,pinfr,
                      xmin,xmax,ymin,ymax,show_phys,show_unphys):
        ql = [rhol, ul, pl]
        qr = [rhor, ur, pr]
        paramsl = [gammal, pinfl]
        paramsr = [gammar, pinfr]
        
        #update_q_values(variable, q1, q2, q3)
        hugoloc1 = lambda p : hugoniot_loci[0](p,ql,paramsl) 
        hugoloc3 = lambda p : hugoniot_loci[1](p,qr,paramsr)
        intcurv1 = lambda p : integral_curves[0](p,ql,paramsl) 
        intcurv3 = lambda p : integral_curves[1](p,qr,paramsr)
        
        # Check whether the 1-wave is a shock or rarefaction
        def phi_l(p):
            global wave1
            if p >= pl:
                wave1 = 'shock'
                return hugoloc1(p)
            else: 
                wave1 = 'rarefaction'
                return intcurv1(p)

        # Check whether the 3-wave is a shock or rarefaction
        def phi_r(p):
            global wave3
            if p >= pr: 
                wave3 = 'shock'
                return hugoloc3(p)
            else: 
                wave3 = 'rarefaction'
                return intcurv3(p)

        phi = lambda p : phi_l(p)-phi_r(p)
        
        # Use fsolve to find p_star such that Phi(p_star)=0
        p0 = (ql[2] + qr[2])/2.0 # initial guess is the average of initial pressures
        p_star, info, ier, msg = fsolve(phi, p0, full_output=True, xtol=1.e-14)
        # For strong rarefactions, sometimes fsolve needs help
        if ier != 1:
            p_star, info, ier, msg = fsolve(phi, p0, full_output=True, factor=0.1, xtol=1.e-10)
            # This should not happen:
            if ier != 1:
                print('Warning: fsolve did not converge.')
        u_star = 0.5*(phi_l(p_star) + phi_r(p_star))

    
        # Set plot bounds
        fig, ax = plt.subplots(figsize=(12,4))
        x = (ql[2] , qr[2], p_star)
        y = (ql[1], qr[1], u_star)
        #xmin, xmax = min(x), max(x)
        #ymin, ymax = min(y), max(y)
        dx, dy = xmax - xmin, ymax - ymin
        #ax.set_xlim(xmin - 0.1*dx, xmax + 0.1*dx)
        #ax.set_ylim(ymin - 0.1*dy, ymax + 0.1*dy)
        ax.set_xlim(min(0.00000001,xmin),xmax)
        ax.set_ylim(ymin,ymax)
        ax.set_xlabel('Pressure (p)', fontsize =15)
        ax.set_ylabel('Velocity (u)', fontsize =15)
        
        #p1 = np.linspace(min(ql[2],p_star),max(ql[2],p_star), 20)
        #p2 = np.linspace(min(p_star,qr[2]),max(p_star, qr[2]), 20)
        p = np.linspace(xmin,xmax,500)
        p1_shk = p[p>=pl]
        p1_rar = p[p<pl]
        p3_shk = p[p>=pr]
        p3_rar = p[p<pr]
        
        #Plot unphyisical solutions
        if show_unphys:
            hugol1_un = hugoloc1(p1_rar)
            hugol3_un = hugoloc3(p3_rar)
            intcu1_un = intcurv1(p1_shk)
            intcu3_un = intcurv3(p3_shk)
            ax.plot(p1_rar,hugol1_un,'--r')
            ax.plot(p3_rar,hugol3_un,'--r')
            ax.plot(p1_shk,intcu1_un,'--b')
            ax.plot(p3_shk,intcu3_un,'--b')
        
        # Plot physical solutions
        if show_phys:
            hugol1_ph = hugoloc1(p1_shk)
            hugol3_ph = hugoloc3(p3_shk)
            intcu1_ph = intcurv1(p1_rar)
            intcu3_ph = intcurv3(p3_rar)
            ax.plot(p1_shk,hugol1_ph,'-r')
            ax.plot(p3_shk,hugol3_ph,'-r')
            ax.plot(p1_rar,intcu1_ph,'-b')
            ax.plot(p3_rar,intcu3_ph,'-b')
            if (p_star <= xmax and u_star >ymin and u_star < ymax):
                ax.plot(p_star, u_star, '-ok', markersize = 10)
                ax.text(x[2] + 0.025*dx,y[2] + 0.025*dy, '$q_m$', fontsize =15)
        
        # Plot initial states and markers
        ax.plot(ql[2], ql[1], '-ok', markersize = 10)
        ax.plot(qr[2], qr[1], '-ok', markersize = 10)
        for i,label in enumerate(('$q_l$', '$q_r$')):
            ax.text(x[i] + 0.025*dx,y[i] + 0.025*dy,label, fontsize =15)
        plt.show()
    return plot_function


# Create the GUI and output the interact app
def interactive_phase_plane(ql=None,qr=None,paramsl=None,paramsr=None):
        # Define initial parameters 
        if ql ==None:	
                ql = [600.0, 10.0, 50000.0]
        if qr == None:
                qr = [50.0, -10.0, 25000.0]
        if paramsl == None:
                paramsl = [1.4, 0.0]
        if paramsr == None:
                paramsr = [7.0, 100.0]


        # Creat plot function for interact
        hugoniot_loci = [hugoniot_locus_1, hugoniot_locus_3]
        integral_curves = [integral_curve_1, integral_curve_3]
        pp_plot = phase_plane_plot(hugoniot_loci, integral_curves)

        # Declare all widget sliders
        ql1_widget = widgets.FloatSlider(value=ql[0],min=0.01,max=1000.0, description=r'$\rho_l$')
        ql2_widget = widgets.FloatSlider(value=ql[1],min=-15,max=15.0, description='$u_l$')
        ql3_widget = widgets.FloatSlider(value=ql[2],min=1,max=200000.0, description='$p_l$')
        qr1_widget = widgets.FloatSlider(value=qr[0],min=0.01,max=1000.0, description=r'$\rho_r$')
        qr2_widget = widgets.FloatSlider(value=qr[1],min=-15,max=15.0, description='$u_r$')
        qr3_widget = widgets.FloatSlider(value=qr[2],min=1,max=200000.0, description='$p_r$')
        gamml_widget = widgets.FloatSlider(value=paramsl[0],min=0.01,max=10.0, description='$\gamma_l$')
        gammr_widget = widgets.FloatSlider(value=paramsr[0],min=0.01,max=10.0, description='$\gamma_r$')
        pinfl_widget = widgets.FloatSlider(value=paramsl[1],min=0.0,max=300000.0, description='$p_{\infty l}$')
        pinfr_widget = widgets.FloatSlider(value=paramsr[1],min=0.0,max=300000.0, description='$p_{\infty r}$')
        xmin_widget = widgets.BoundedFloatText(value=0.0000001, description='Xmin:')
        xmax_widget = widgets.FloatText(value=200000, description='Xmax:')
        ymin_widget = widgets.FloatText(value=-15, description='Ymin:')
        ymax_widget = widgets.FloatText(value=15, description='Ymax:')
        show_physical = widgets.Checkbox(value=True, description='Physical solution')
        show_unphysical = widgets.Checkbox(value=True, description='Unphysical solution')
        # Additional control widgets not called by function
        rhomax_widget = widgets.FloatText(value=1000, description=r'$\rho_{max}$')
        gammax_widget = widgets.FloatText(value=10, description='$\gamma_{max}$')
        pinfmax_widget = widgets.FloatText(value=300000, description='$p_{\infty max}$')

        # Allow for dependent widgets to update
        def update_xmin(*args):
            ql3_widget.min = xmin_widget.value
            qr3_widget.min = xmin_widget.value
        def update_xmax(*args):
            ql3_widget.max = xmax_widget.value
            qr3_widget.max = xmax_widget.value
        def update_ymin(*args):
            ql2_widget.min = ymin_widget.value
            qr2_widget.min = ymin_widget.value
        def update_ymax(*args):
            ql2_widget.max = ymax_widget.value
            qr2_widget.max = ymax_widget.value
        def update_rhomax(*args):
            ql1_widget.max = rhomax_widget.value
            qr1_widget.max = rhomax_widget.value
        def update_gammax(*args):
            gamml_widget.max = gammax_widget.value
            gammr_widget.max = gammax_widget.value
        def update_pinfmax(*args):
            pinfl_widget.max = pinfmax_widget.value
            pinfr_widget.max = pinfmax_widget.value
        xmin_widget.observe(update_xmin, 'value')
        xmax_widget.observe(update_xmax, 'value')
        ymin_widget.observe(update_ymin, 'value')
        ymax_widget.observe(update_ymax, 'value')
        rhomax_widget.observe(update_rhomax, 'value')
        gammax_widget.observe(update_gammax, 'value')
        pinfmax_widget.observe(update_pinfmax, 'value')


        # Organize slider widgets into boxes
        qleft = widgets.HBox([ql1_widget, ql2_widget, ql3_widget])
        qright = widgets.HBox([qr1_widget, qr2_widget, qr3_widget])
        params = widgets.HBox([widgets.VBox([gamml_widget, gammr_widget]), 
	                       widgets.VBox([pinfl_widget , pinfr_widget])])
        plot_opts = widgets.HBox([widgets.VBox([show_physical, xmin_widget , xmax_widget]),
	                         widgets.VBox([show_unphysical, ymin_widget , ymax_widget]),
	                         widgets.VBox([rhomax_widget, gammax_widget, pinfmax_widget])])

        # Set up interactive GUI
        interact_gui = widgets.Accordion(children=[qleft, qright, params, plot_opts])
        interact_gui.set_title(0, 'Left state')
        interact_gui.set_title(1, 'Right state')
        interact_gui.set_title(2, 'Tammann EOS parameters')
        interact_gui.set_title(3, 'Plot options')


        # Define interactive widget and run GUI
        ppwidget = interact(pp_plot, rhol = ql1_widget, ul = ql2_widget, pl = ql3_widget,
	                          rhor = qr1_widget, ur = qr2_widget, pr = qr3_widget,
	                          gammal = gamml_widget, pinfl = pinfl_widget,
	                          gammar = gammr_widget, pinfr = pinfr_widget,
	                          xmin = xmin_widget, xmax = xmax_widget,
	                          ymin = ymin_widget, ymax = ymax_widget,
	                          show_phys = show_physical, show_unphys = show_unphysical)
        ppwidget.widget.close()
        display(interact_gui)
        display(ppwidget.widget.out)
