# Riemann_book
Book in progress to illustrate Riemann solvers in Jupyter notebooks.
@rjleveque, @ketch, and @maojrs.

We will start by collecting some things from the following repositories to work on: 
  clawpack/riemann, clawpack/apps/notebooks, and clawpack/clawutil
  
##To do:

- [x] ~~Add dropdown to static interacts to select variable~~ Vertical layout instead.
- [x] Phase space plots where you can drag q_l and q_r around
- [x] Sliders for sample q_l and q_r inputs (in phase space, also in x-t plane?)
- [x] Linked phase plane, q(x, t=1), and/or x-t plane plots as q_l, q_r change
- [ ] Incorporate interactive general linear phase plane as function in riemann_tools.py
- [ ] Create "riemann_apps.py" to store functions for specific interactive apps, like the one
for shallow water?

##Early chapters
Should the book start with some general chapters explaining important background?  Or just jump into some simple hyperbolic systems and explain the concepts as they are encountered?  Some of the things that need to be explained are:
- Similarity solutions
- Characteristics and characteristic velocities
- Conservation, weak solutions, and jump conditions
- Riemann invariants and integral curves
- How approximate solvers are used in numerical discretizations

##Problems

###One-dimensional

- [ ] Advection - conservative and color equation
- [ ] Acoustics - constant coefficient and arbitrary rho, K on each side
- [ ] Burgers' - with/without entropy fix
- [ ] Traffic flow - scalar and systems
- [ ] Buckley-Leverett
- [ ] Shallow water - Exact, Roe, HLLE  (and with tracer)
- [ ] Shallow water with topography, Augmented solver
- [ ] p-system / nonlinear elasticity
- [ ] Euler - Exact, Roe, HLL, HLLE, HLLC, 
- [ ] Euler with general EOS
- [ ] Reactive Euler
- [ ] HLL solver for arbitrary equations
- [ ] Layered shallow water
- [ ] MHD
- [ ] Relativistic Euler
- [ ] Dusty gas
- [ ] Two-phase flow

###Two-dimensional

- [ ] Elasticity
- [ ] Maxwell's equations
- [ ] Arbitrary normal direction on mapped grid
- [ ] Poro-elasticity (?)

## What each chapter should contain (optional things in *italics*)
- Description of the equations 
- *physical derivation*
- Analysis of the hyperbolic structure: 
	- Jacobian; eigenvalues and eigenvectors
	- Rankine-Hugoniot jump conditions
	- Riemann invariants
	- *structure of centered rarefaction waves*
- Riemann solvers
	- Exact Riemann solver
	- *Approximate Riemann solvers*
	- *Solvers for mapped grids*
	- *Well-balanced solvers incorporating source terms*
	- *Solvers with and without entropy fix*
	- *Discussion and solvers for the transverse problem*
	- *Comparisons*
- *Results using Clawpack with different solvers*

## Notebooks already written that should be adapted as chapters in the book
- [Acoustics, including transverse solver](http://nbviewer.ipython.org/github/maojrs/ipynotebooks/blob/master/acoustics_riemann.ipynb)
- [Elasticity, including transverse solver](http://nbviewer.ipython.org/github/maojrs/ipynotebooks/blob/master/elasticity_riemann.ipynb)
- [Shallow water equations](http://nbviewer.ipython.org/url/faculty.washington.edu/rjl/notebooks/shallow/SW_riemann_tester.ipynb)
- [Euler equations (exact solution only)](http://nbviewer.ipython.org/gist/ketch/08ce0845da0c8f3fa9ff)
