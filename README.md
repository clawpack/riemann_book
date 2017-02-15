# Riemann_book
Book in progress to illustrate Riemann solvers in Jupyter notebooks.
@rjleveque, @ketch, and @maojrs.

We will start by collecting some things from the following repositories to work on: 
  clawpack/riemann, clawpack/apps/notebooks, and clawpack/clawutil
  
## To do:

- [x] ~~Add dropdown to static interacts to select variable~~ Vertical layout instead.
- [x] Phase space plots where you can drag q_l and q_r around
- [x] Sliders for sample q_l and q_r inputs (in phase space, also in x-t plane?)
- [x] Linked phase plane, q(x, t=1), and/or x-t plane plots as q_l, q_r change
- [x] Incorporate interactive general linear phase plane as function in riemann_tools.py
- [x] Create "riemann_apps.py" to store functions for specific interactive apps, like the one
for shallow water?

## Early chapters
Should the book start with some general chapters explaining important background?  Or just jump into some simple hyperbolic systems and explain the concepts as they are encountered?  Some of the things that need to be explained are:
- Similarity solutions
- Characteristics and characteristic velocities
- Conservation, weak solutions, and jump conditions
- Riemann invariants and integral curves
- How approximate solvers are used in numerical discretizations

## Problems

### One-dimensional

- [ ] **Advection** - conservative and color equation
- [x] Acoustics - constant coefficient and arbitrary rho, K on each side - Mauricio
- [x] Traffic flow - scalar - David
- [ ] **Burgers'** - with/without entropy fix - Mauricio
- [x] Buckley-Leverett - Randy
- [ ] **Shallow water** - Exact, Roe, HLLE  (and with tracer) - Randy
- [ ] **Shallow water with topography**, Augmented solver - Randy
- [x] p-system / nonlinear elasticity - David
- [x] Euler - Exact, Roe, HLL, HLLE, HLLC 
- [ ] **Euler with general EOS** - Mauricio
- [ ] Reactive Euler
- [ ] Traffic - systems
- [ ] LLF and HLL solvers for arbitrary equations
- [ ] Layered shallow water - David
- [ ] MHD
- [ ] Relativistic Euler
- [ ] Dusty gas
- [ ] Two-phase flow

### Two-dimensional

- [ ] Elasticity - Mauricio
- [ ] Acoustics + mapped grids - Mauricio
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


# Citations

We are using bibtex for citations; add entries as necessary to `riemann.bib`.
To insert a citation in a notebook, follow this pattern:

    <cite data-cite="toro2013riemann"><a href="riemann.html#toro2013riemann">(Toro, 2013)<a></cite>

The value appearing in the html tag and the hyperlink should match the cite key
in the bibtex file.

## How to generate the html bibliography

Use bibtex2html, downloadable from https://www.lri.fr/~filliatr/bibtex2html/.
Then:

    export TMPDIR=.
    bibtex2html riemann.bib
    
This creates riemann.html which includes an anchor for each citation.

We are keeping the HTML file under version control for convenience, even
though it is generated from the .bib file.


## How to generate the PDF with bibliography

I've tested this with Jupyter 5.2.

    jupyter nbconvert --to latex --template citations.tplx Euler_equations.ipynb
    pdflatex Euler_equations
    bibtex Euler_equations
    pdflatex Euler_equations

This generates a single chapter with a bibliography at the end.  To actually
generate the book, we'll want to merge all the notebooks first; see
discussion at https://github.com/jupyter/nbconvert/issues/253.
