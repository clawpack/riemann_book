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
To install the dependencies for the book, see
https://github.com/clawpack/riemann_book/wiki/Installation.  Then clone this
repository to get all the notebooks.  A table of contents and suggested order
for reading the notebooks is given in `Index.ipynb`.

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

### Windows Azure

Rather than installing software, you can execute the notebooks on the cloud
using the [Microsoft Azure Notebooks](https://notebooks.azure.com) cloud
service:  Create a free account and then clone the [riemann_book
library](https://notebooks.azure.com/rjleveque/libraries/riemann-book).
*These may not be up to date with the versions in this repository during the
development phase of this project.*

### Binder

This is still under development using the latest version of
[binder](https://beta.mybinder.org/).  You can try it out for these notebooks
at this link: https://beta.mybinder.org/v2/gh/clawpack/riemann_book/master

This should start up a notebook server on a
[Jupyterhub](https://jupyterhub.readthedocs.io/en/latest/) that lets you
execute all the notebooks with no installation required.
