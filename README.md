[![Build Status](https://travis-ci.org/clawpack/riemann_book.svg?branch=master)](https://travis-ci.org/clawpack/riemann_book)

# Riemann_book
Book in progress to illustrate Riemann solvers in Jupyter notebooks.
Contributors: @rjleveque, @ketch, and @maojrs.
  
# Dependencies
- matplotlib
- numpy
- clawpack, including the bleeding edge version of clawpack.riemann
- jupyter
- ipywidgets
- mpl_toolkits

# Installation
First, install a Fortran compiler.  Then:

```
git clone --branch=master --depth=100 --quiet git://github.com/clawpack/clawpack
cd clawpack
git submodule init
git submodule update clawutil visclaw riemann
cd riemann
git checkout master
git pull # get very latest Riemann
cd ..
python setup.py install
cd ..
git clone https://github.com/clawpack/riemann_book
cd riemann_book
pip install -r requirements.txt
jupyter nbextension enable --py widgetsnbextension
```

You can test your installation by running

```
python test.py
```

## Outline
Parentheticals indicate concepts introduced for the first time.

- [Prologue: background and motivation](https://github.com/clawpack/riemann_book/wiki/Introductory-notebook-outline)
1. [Acoustics](https://github.com/clawpack/riemann_book/blob/master/Acoustics.ipynb) (eigenvalue analysis, characteristics, similarity solutions)
2. [Traffic flow](https://github.com/clawpack/riemann_book/blob/master/Traffic_flow.ipynb) (shocks, rarefactions, conservation, jump conditions)
3. Burgers' (weak solutions, approximate solvers, entropy fix)
4. [Shallow water](https://github.com/clawpack/riemann_book/blob/master/Shallow_tracer.ipynb) (jump conditions for a nonlinear system; Riemann invariants, integral curves, Hugoniot Loci; Roe solver; HLL solver) (see also [this](http://nbviewer.jupyter.org/url/faculty.washington.edu/rjl/notebooks/shallow/SW_riemann_tester.ipynb) and [this](http://nbviewer.jupyter.org/gist/rjleveque/8994740)
5. [Euler equations](https://github.com/clawpack/riemann_book/blob/master/Euler_equations.ipynb) (contact waves)
6. [Euler approximate solvers](https://github.com/clawpack/riemann_book/blob/master/Euler_approximate_solvers.ipynb) (more in-depth look at Roe and HLL solvers; could just be combined with previous chapter)
7. [Euler with Tamman EOS](https://github.com/clawpack/riemann_book/blob/master/Euler_equations_TammannEOS.ipynb)
8. Shallow water with topography (source terms)
9. The "kitchen sink" problem (geometric source terms; non-uniform left/right states)
10. Traffic flow with a variable speed limit (spatially-varying flux)
11. [Nonlinear elasticity](https://github.com/clawpack/riemann_book/blob/master/Nonlinear_elasticity.ipynb) (f-wave approximate solvers)
12. [Non-convex flux](https://github.com/clawpack/riemann_book/blob/master/Nonconvex_Scalar_Osher_Solution.ipynb): Buckley-Leverett, night-time traffic
13. [2D Acoustics](http://nbviewer.jupyter.org/github/maojrs/ipynotebooks/blob/master/acoustics_riemann.ipynb) (2D, transverse solvers)
14. [2/3D Elasticity](http://nbviewer.jupyter.org/github/maojrs/ipynotebooks/blob/master/elasticity_riemann.ipynb) (3D)
15. 2D Euler (do quadrants problem, discuss 2D Riemann problems)

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
- [ ] **Shallow water in cylindrical coordinates** - David
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
