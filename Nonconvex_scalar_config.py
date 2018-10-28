#import seaborn as sns
#sns.set_style('white',{'legend.frameon':'True'});
import matplotlib as mpl
mpl.rcParams['font.size'] = 10
figsize =(8,4)
mpl.rcParams['figure.figsize'] = figsize
mpl.rcParams['figure.dpi'] = 100
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import widgets, fixed
from ipywidgets import interact
from utils import riemann_tools
from exact_solvers import nonconvex
from exact_solvers import nonconvex_demos
