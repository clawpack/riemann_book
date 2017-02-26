
# Makefile for Clawpack code in this directory.
# This version only sets the local files and frequently changed
# options, and then includes the standard makefile pointed to by CLAWMAKE.
CLAWMAKE = $(CLAW)/clawutil/src/Makefile.common

# See the above file for details and a list of make options, or type
#   make .help
# at the unix prompt.


# Adjust these variables if desired:
# ----------------------------------

CLAW_PKG = classic                  # Clawpack package to use
EXE = xclaw                         # Executable to create
SETRUN_FILE = setrun.py             # File containing function to make data
OUTDIR = _output                    # Directory for output
SETPLOT_FILE = setplot.py           # File containing function to set plots
PLOTDIR = _plots                    # Directory for plots

OVERWRITE ?= True                   # False ==> make a copy of OUTDIR first
RESTART ?= False                    # Should = clawdata.restart in setrun

# Environment variable FC should be set to fortran compiler, e.g. gfortran

# Compiler flags can be specified here or set as an environment variable
FFLAGS ?=  

# ---------------------------------
# List of sources for this program:
# ---------------------------------

MODULES = \

SOURCES = \
  qinit.f \
  setprob.f \
  rp1_shallow_roe_tracer.f90 \
  $(CLAW)/classic/src/1d/setaux.f \
  $(CLAW)/classic/src/1d/bc1.f \
  $(CLAW)/classic/src/1d/b4step1.f \
  $(CLAW)/classic/src/1d/driver.f90 \
  $(CLAW)/classic/src/1d/claw1ez.f \
  $(CLAW)/classic/src/1d/claw1.f \
  $(CLAW)/classic/src/1d/copyq1.f \
  $(CLAW)/classic/src/1d/inlinelimiter.f90 \
  $(CLAW)/classic/src/1d/opendatafile.f \
  $(CLAW)/classic/src/1d/out1.f \
  $(CLAW)/classic/src/1d/src1.f \
  $(CLAW)/classic/src/1d/step1.f90

#-------------------------------------------------------------------
# Include Makefile containing standard definitions and make options:
include $(CLAWMAKE)

  #$(CLAW)/riemann/src/rp1_shallow_tracer.f90 \
