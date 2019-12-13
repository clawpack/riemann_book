[![Build Status](https://travis-ci.org/clawpack/riemann_book.svg?branch=master)](https://travis-ci.org/clawpack/riemann_book)

# Riemann Problems and Jupyter Solutions

#### by David I. Ketcheson, Randall J. LeVeque, and Mauricio del Razo Sarmina

This repository contains work on a book in progress (nearly complete) to illustrate Riemann
solutions and approximate Riemann solvers in Jupyter notebooks.

Contributors: @ketch, @rjleveque, and @maojrs.

## License

### Code

The code in this repository, including all code samples in the notebooks,
is released under the 3-Clause BSD License.  See
[LICENSE-CODE](https://github.com/clawpack/riemann_book/blob/master/LICENSE-CODE)
for the license and read more at the 
[Open Source Initiative](https://opensource.org/licenses/bsd-3-clause).

### Text

The text content of the notebooks is released under the CC-BY-NC-ND License.
See
[LICENSE-TEXT.md](https://github.com/clawpack/riemann_book/blob/master/LICENSE-TEXT.md)
for the license and read more at [Creative
Commons](https://creativecommons.org/licenses/by-nc-nd/4.0/).



## View static webpages

The notebooks are saved in Github with the output stripped out.  You can view
the html rendered notebooks with output intact [on this
webpage](http://www.clawpack.org/riemann_book/index.html).  These are static
views (no execution or interactive widgets), but some notebooks include
animations that will play.  *These may not be up to date with the versions in
this repository during the development phase of this project.*

## Installation
To install the dependencies for the book, first install a Fortran compiler.
Then do the following in a terminal:

```
pip install clawpack
git clone https://github.com/clawpack/riemann_book
cd riemann_book
pip install -r requirements.txt
jupyter nbextension enable --py widgetsnbextension
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable equation-numbering/main
```

You can test your installation by running

```
python test.py
```

A table of contents and suggested order for reading the notebooks is given in `Index.ipynb`.

If you want to compile the PDF locally, you must also install the package `bookbook`.

## Docker

Rather than installing all the dependencies, if you have
[Docker](https://www.docker.com/) installed you can use

    $ docker pull clawpack/rbook

to obtain a docker image that has all the notebooks and dependencies
installed.  This was built using the `Dockerfile` in
this repository, which could be modified to build a new image also
containing other material, if desired.  See `Docker.md` for further
instructions.

## Execute in the cloud

### Binder

Rather than installing anything on your own computer, you can run the
notebooks on the cloud using the free
[binder](https://mybinder.org/) service.  
Simply navigate to this link in a browser:

https://mybinder.org/v2/gh/clawpack/riemann_book/master

This may take a few minutes to start up a notebook server on a
[Jupyterhub](https://jupyterhub.readthedocs.io/en/latest/). Then navigate to
`riemann_book` and open `Index.ipynb` to get started.
