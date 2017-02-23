# Riemann_book
Book in progress to illustrate Riemann solvers in Jupyter notebooks.
Contributors: @rjleveque, @ketch, and @maojrs.
  

## Early chapters
Should the book start with some general chapters explaining important background?  Or just jump into some simple hyperbolic systems and explain the concepts as they are encountered?  Some of the things that need to be explained are:
- Similarity solutions
- Characteristics and characteristic velocities
- Conservation, weak solutions, and jump conditions
- Riemann invariants and integral curves
- How approximate solvers are used in numerical discretizations

## Chapters
Chapters with a complete draft have the box checked.  Chapters that are required are in bold.  The remaining chapters are optional and will depend on the authors finding time to complete them.

### One-dimensional

- [ ] **Advection** - conservative and color equation
- [x] **Acoustics** - constant coefficient and arbitrary rho, K on each side - Mauricio
- [x] **Traffic flow** - scalar - David
- [x] Traffic flow with variable speed limit - David
- [ ] **Burgers'** - with/without entropy fix - Mauricio
- [x] **Buckley-Leverett** - Randy
- [ ] **Shallow water** - Exact, Roe, HLLE  (and with tracer) - Randy
- [ ] **Shallow water with topography**, Augmented solver - Randy
- [x] **p-system / nonlinear elasticity** - David
- [x] **Euler** - Exact, Roe, HLL, HLLE, HLLC 
- [x] **Euler with general EOS** - Mauricio
- [ ] Reactive Euler
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
