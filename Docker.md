
# To use Docker

**Note:** In the instructions below, 
 - `$` refers to the native bash shell prompt on your computer, 
 - `root@...#` refers to the bash shell prompt within the Docker container, once you have that running.

First install [Docker](https://www.docker.com/).

Then, in this directory (containing the file `Dockerfile`), do:

    $ docker build -t riemann_book_dockerimage -f Dockerfile .

Don't forget the last `.` on this line!

Then do:

    $ docker run -i -t -p 8889:8889 --name riemann_book_container riemann_book_dockerimage

This starts a virtual machine (*container*) named `riemann_book_container` and gives a prompt like: 

    root@...# 

### Updating the riemann_book files

In case the `riemann_book` repository changed since you built the docker image, you could do:

    root@...# cd /riemann_book
    root@...# git pull
    root@...# cd /
    
### Updating `clawpack/riemann`

You may need some Riemann solvers not in the most recent release of Clawpack.  These can be obtained by checking out the master branch (and pulling any changes since you built the image, if necessary):

    root@...# cd /clawpack-5.4.0/riemann
    root@...# git checkout master
    root@...# git pull

If this brings down new Riemann solvers, you will need to compile them and re-install clawpack:

    root@...# cd /clawpack-5.4.0
    root@...# pip install -e .
    
### Test our WIP `Shallow_water` notebook 

    root@...# cd /riemann_book
    root@...# git remote add ketch https://github.com/ketch/riemann_book
    root@...# git fetch ketch
    root@...# git checkout ketch/shallow_water
    root@...# cd /
    
### Notebook server:

In order to work with the notebooks, start the notebook server via

    root@...# jupyter notebook --notebook-dir=/riemann_book --ip='*' --port=8889 --no-browser

Then open a browser (on your laptop) to the URL printed out when the Jupyter server starts via the command above.  This might be of the form:

    http://localhost:8889/tree
    
or perhaps will include a token, something like:

    http://localhost:8889/?token=9542fb1ed940f873f28e6a371c6334c5b1a0d8656121905c
    
This should open a web page with the list of all available notebooks.  You can start with `Index.ipynb`, or click on any notebook to launch.

Use `ctrl-C` to exit the Jupyter notebook server. 

See http://jupyter.org/ for more documentation on Jupyter.

### Connecting with a second bash shell

If you have the notebook server running and also want another window open with a bash shell, in another shell on your laptop you can do:

    $ docker exec -it riemann_book_container bash
    
### Exiting a shell / halting a container

Use `Ctrl-p Ctrl-q` to exit from a shell without halting the docker container.

You can halt the container (after using `ctrl-C` to quit the jupyter server if
one is running) via::

    root@...# exit

### Restarting a container

You can restart the container via::

    $ docker start -a -i riemann_book_container

The external port should still work for serving notebooks.

### Removing a container

This gets rid of the container and any data that you might have created when running this container:

    $ docker rm riemann_book_container
    
### Removing the image

You might want to do this if you need to rebuild an image with an updated Dockerfile or to incorporate a newer version of software:

    $ docker rmi riemann_book_dockerimage
    
### More resources:

 - [Docker for Clawpack](http://www.clawpack.org/docker_image.html#docker-image).  The Dockerfile provided here includes everything in the Clawpack-5.4.0 image.
 
 - [Introduction to Docker](https://geohackweek.github.io/Introductory/01-docker-tutorial/) from 
   the recent [GeoHack Week](https://geohackweek.github.io) at UW.
