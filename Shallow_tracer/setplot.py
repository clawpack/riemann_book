
""" 
Set up the plot figures, axes, and items to be done for each frame.

This module is imported by the plotting routines and then the
function setplot is called to set the plot parameters.
    
""" 

from pylab import *

from clawpack.clawutil.data import ClawData
setprob_data = ClawData()
setprob_data.read('setprob.data', force=True)

#--------------------------
def setplot(plotdata):
#--------------------------
    
    """ 
    Specify what is to be plotted at each frame.
    Input:  plotdata, an instance of clawpack.visclaw.data.ClawPlotData.
    Output: a modified version of plotdata.
    
    """ 

    from clawpack.visclaw.colormaps import yellow_red_blue as yrb

    plotdata.clearfigures()  # clear any old figures,axes,items data

    def compute_approx_Rsoln(current_data):
        hl = setprob_data.hl
        hr = setprob_data.hr
        ul = setprob_data.ul
        ur = setprob_data.ur
        g = setprob_data.g
        s1left = ul - sqrt(g*hl)
        s2right = ur + sqrt(g*hl)
        hbar = 0.5*(hl+hr)
        uhat = (sqrt(hl)*ul + sqrt(hr)*ur) / (sqrt(hl) + sqrt(hr)) 
        s1roe = uhat - sqrt(g*hbar)
        s2roe = uhat + sqrt(g*hbar)

        solver = 'roe'
        if solver == 'onesided':
            s1 = s1left
            s2 = s2right
        elif solver == 'roe':
            s1 = s1roe
            s2 = s2roe
        elif solver == 'hlle':
            s1 = min(s1roe, s1left)
            s2 = max(s2roe, s2right)

        hm = (hl*ul - hr*ur + s2*hr - s1*hl) / (s2 - s1)
        current_data.user['s1'] = s1
        current_data.user['s2'] = s2
        current_data.user['hm'] = hm


    def plot_approx_Rsoln(current_data):
        from pylab import plot
        x = current_data.x
        t = current_data.t
        h = current_data.q[:,0]
        s1 = current_data.user['s1']
        s2 = current_data.user['s2']
        hm = current_data.user['hm']
        x1 = s1*t
        x2 = s2*t
        xvec = [x[0],x1,x1,x2,x2,x[-1]]
        hvec = [h[0],h[0],hm,hm,h[-1],h[-1]]
        plot(xvec, hvec, 'k-', linewidth=2)
        
    #plotdata.beforeframe = compute_approx_Rsoln

    # Figure for q[0]
    plotfigure = plotdata.new_plotfigure(name='Depth', figno=1)
    plotfigure.kwargs = {'figsize': [6,6]}

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.axescmd = 'subplot(211)'
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = [-0.1,3.5]
    plotaxes.title = 'Depth and Tracer'

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = 0
    plotitem.color = 'k'
    plotitem.kwargs = {'linewidth':2}
    def fill_with_cmap(current_data):
        from clawpack.visclaw.colormaps import make_colormap
        import string
        x = current_data.x
        h = current_data.q[0,:]
        u = current_data.q[1,:] / h
        tracer = current_data.q[2,:]
        X = vstack((x,x))
        Y = vstack((0*x, h))
        c = 0.*x
        i1 = array([mod(int(12*tracer[i]),2) for i in range(len(x))])
        c = where((i1==0) & (tracer>0), 1, c)
        c = where((i1==0) & (tracer<0), -1, c)
        c = where((i1==1) & (tracer<0), -0.5, c)
        c = where((i1==1) & (tracer>0), 0.5, c)
        cmin = -1.0
        cmax = 1.0
        cmid = 0.5*(cmin+cmax)
        cmap = make_colormap({-1: [0.9,0.9,1.0], \
                              -0.001:[0.0,0.0,1.0], \
                               0.001:[1.0,0.4,0.4], \
                              1.0:[1.0,0.9,0.9]})
        cmap = make_colormap({cmin: [1.0,0.0,0.0], -0.5:[1.,.8,.8],
                    cmid:[.8,1,1.], cmax:[0.,0.,1]})

        C = vstack((c,c))
        pcolor(X,Y,C,cmap=cmap)
        outdir = current_data.plotdata.outdir
        frameno = string.zfill(current_data.frameno,4)
        #plot_approx_Rsoln(current_data)
        ylim([0,3.5])
        #axis('off')

    plotaxes.afteraxes = fill_with_cmap

    plotaxes = plotfigure.new_plotaxes('velocity')
    plotaxes.axescmd = 'subplot(212)'
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = [-1.0, 1.0]
    plotaxes.title = 'Velocity'

    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    def velocity(current_data):
        h = current_data.q[0,:]
        hu = current_data.q[1,:]
        u = hu / h
        return u
    plotitem.plot_var = velocity
    plotitem.color = 'k'
    plotitem.kwargs = {'linewidth':2}

    # Figure for tracer
    plotfigure = plotdata.new_plotfigure(name='Tracer', figno=5)
    plotfigure.show = False
    plotfigure.kwargs = {'figsize': [10,3]}

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = [-1.5,1.5]
    plotaxes.title = 'Tracer'

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='1d_plot')
    plotitem.plot_var = 2
    plotitem.plotstyle = '-'
    plotitem.color = 'b'

    # Parameters used only when creating html and/or latex hardcopy
    # e.g., via clawpack.visclaw.frametools.printframes:

    plotdata.printfigs = True                # print figures
    plotdata.print_format = 'png'            # file format
    plotdata.print_framenos = 'all'          # list of frames to print
    plotdata.print_fignos = 'all'            # list of figures to print
    plotdata.html = True                     # create html files of plots?
    plotdata.latex = True                    # create latex file of plots?
    plotdata.latex_figsperline = 2           # layout of plots
    plotdata.latex_framesperline = 1         # layout of plots
    plotdata.latex_makepdf = False           # also run pdflatex?

    return plotdata

    
