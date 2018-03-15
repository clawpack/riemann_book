[![Build Status](https://travis-ci.org/clawpack/riemann_book.svg?branch=master)](https://travis-ci.org/clawpack/riemann_book)

# Riemann_book

## The Riemann Problem for Hyperbolic PDEs: Theory and Approximate Solvers

#### by David I. Ketcheson, Randall J. LeVeque, and Mauricio del Razo Sarmina

This repository contains work on a book in progress to illustrate Riemann
solvers in Jupyter notebooks.

Contributors: @ketch, @rjleveque, and @maojrs.

## View static webpages

The notebooks are saved in Github with the output stripped out.  You can view
the html rendered notebooks with output intact [on this
webpage](http://www.clawpack.org/riemann_book/index.html).  These are static
views (no execution or interactive widgets), but some notebooks include
animations that will play.  *These may not be up to date with the versions in
this repository during the development phase of this project.*

# Installation
To install the dependencies for the book, see
https://github.com/clawpack/riemann_book/wiki/Installation.  Then clone this
repository to get all the notebooks.  A table of contents and suggested order
for reading the notebooks is given in `Index.ipynb`.

## Docker

Rather than installing all the dependencies, if you have
[Docker](https://www.docker.com/) installed you can use the `Dockerfile` in
this repository.  See `Docker.md` for instructions.

*[Add instructions for Dockerhub]*

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
