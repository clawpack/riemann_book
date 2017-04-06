[![Build Status](https://travis-ci.org/clawpack/riemann_book.svg?branch=master)](https://travis-ci.org/clawpack/riemann_book)

# Riemann_book
Book in progress to illustrate Riemann solvers in Jupyter notebooks.
Contributors: @rjleveque, @ketch, and @maojrs.

# Installation
To install the dependencies for the book, see https://github.com/clawpack/riemann_book/wiki/Installation

## Outline
Parentheticals indicate concepts introduced for the first time.

- [Prologue: background and motivation](https://github.com/clawpack/riemann_book/wiki/Introductory-notebook-outline)

**Part I: Exact Riemann solutions**
1. [Advection](https://github.com/clawpack/riemann_book/blob/master/Advection.ipynb)
1. [Acoustics](https://github.com/clawpack/riemann_book/blob/master/Acoustics.ipynb) (eigenvalue analysis, characteristics, similarity solutions)
2. [Traffic flow](https://github.com/clawpack/riemann_book/blob/master/Traffic_flow.ipynb) (shocks, rarefactions, conservation, jump conditions)
3. Burgers' (weak solutions)
4. [Shallow water](https://github.com/clawpack/riemann_book/blob/master/Shallow_tracer.ipynb) (jump conditions for a nonlinear system; Riemann invariants, integral curves, Hugoniot Loci) (see also [this](http://nbviewer.jupyter.org/url/faculty.washington.edu/rjl/notebooks/shallow/SW_riemann_tester.ipynb) and [this](http://nbviewer.jupyter.org/gist/rjleveque/8994740))
5. How to solve the Riemann problem exactly -- go in depth into SW solver, including Newton iteration to find root of piecewise function
5. Shallow water with a tracer (contact waves)
5. [Euler equations](https://github.com/clawpack/riemann_book/blob/master/Euler_equations.ipynb)

**Part II: Approximate solvers**
1. Motivation and approaches to approximate solvers (waves vs fluxes)
1. Transonic rarefactions and entropy fixes
2. Linearized solvers (Roe) (non-physical solutions)
3. LLF and HLL and extensions (smearing of contacts)
4. Comparison of solvers for shallow water
5. Comparison of solvers for Euler
6. Comparison of full numerical solutions for Woodward-Colella blast wave problem

**Part III: Riemann problems in heterogeneous media**
1. Advection (conservative vs color)
1. Acoustics (conservative vs non-conservative)
2. [Variable speed-limit traffic](https://github.com/clawpack/riemann_book/blob/master/Traffic_variable_speed.ipynb)
3. [Nonlinear elasticity](https://github.com/clawpack/riemann_book/blob/master/Nonlinear_elasticity.ipynb) (forward reference to nonconservative nonlinear problems)
4. Shock tube with different ratio of specific heats
7. [Euler with Tamman EOS](https://github.com/clawpack/riemann_book/blob/master/Euler_equations_TammannEOS.ipynb)


**Part IV: Source terms**
1. Approaches: source at interface vs other approaches (not covered here), well-balancing? stiffness?
1. Scalar example(s) (advection-reaction, traffic with on-ramps, burgers-reaction) (well-balancing)
1. Shallow water with bathymetry
2. Euler with gravity
3. Reactive Euler?
4. Discuss viscous source terms?

**Part V: Non-classical problems**
1. [Nonconvex fluxes](https://github.com/clawpack/riemann_book/blob/master/Nonconvex_Scalar_Osher_Solution.ipynb) (Buckley-Leverett, Osher solution)
2. Pressureless gas (non-diagonalizable)
3. (nonconvex flux systems -- MHD?)
4. Nonconservative, nonlinear systems (path-conservative solvers)

**Part VI: Multidimensional systems**
1. Planar Riemann problem for a multi-D system (e.g. [Acoustics](http://nbviewer.jupyter.org/github/maojrs/ipynotebooks/blob/master/acoustics_riemann.ipynb), SW, Euler) (shear waves)
2. [Elasticity](http://nbviewer.jupyter.org/github/maojrs/ipynotebooks/blob/master/elasticity_riemann.ipynb)
3. Quadrants problem (2D Euler Riemann-like problem)
3. Cylindrical shallow water



## Chapters
Chapters with a complete draft have the box checked.  Chapters that are required are in bold.  The remaining chapters are optional and will depend on the authors finding time to complete them.

### One-dimensional

- [x] **Advection** - conservative and color equation
- [x] **Acoustics** - constant coefficient and arbitrary rho, K on each side - Mauricio
- [x] **Traffic flow** - scalar - David
- [x] Traffic flow with variable speed limit - David
- [ ] **Burgers'** - with/without entropy fix - Mauricio
- [x] **Buckley-Leverett** - Randy
- [x] **Shallow water** - Exact, Roe, HLLE  (and with tracer)
- [ ] **Shallow water with topography**, Augmented solver - Randy
- [x] **Shallow water in cylindrical coordinates** - David
- [x] **p-system / nonlinear elasticity** - David
- [x] **Euler** - Exact, Roe, HLL, HLLE, HLLC 
- [x] **Euler with general EOS** - Mauricio
- [ ] Reactive Euler - Luiz Faria wrote some solvers here: https://github.com/ketch/pyclaw-detonation
- [ ] Traffic - systems
- [ ] LLF and HLL solvers for arbitrary equations
- [ ] Layered shallow water - David
- [ ] MHD
- [ ] Relativistic Euler
- [ ] Dusty gas
- [ ] Two-phase flow

### Two-dimensional

- [ ] **Elasticity** - Mauricio
- [ ] **Acoustics + mapped grids** - Mauricio
- [ ] Shallow water
- [ ] Euler
- [ ] Maxwell's equations
- [ ] Arbitrary normal direction on mapped grid
- [ ] Poro-elasticity

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


## Inserting citations
See https://github.com/clawpack/riemann_book/wiki/Citations
