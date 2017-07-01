
# To use Docker

**Note:** In the instructions below, 
 - `$` refers to the native bash shell prompt on your computer, 
 - `root@...#` refers to the bash shell prompt within the Docker container, once you have that running.

First install [Docker](https://www.docker.com/).  If it is installed, make sure it is running.

Then, in this directory (containing the file `Dockerfile`), do:

    $ docker build -t riemann_book_dockerimage .

Don't forget the last `.` on this line!

Then do:

    $ docker run -it -p 8888:8888 --name riemann_book_container riemann_book_dockerimage

This starts a virtual machine (*container*) named `riemann_book_container` and starts a Jupyter Notebook server.
You should see a URL to copy paste into a browser to start your notebook, such as

    http://localhost:8888/?token=9542fb1ed940f873f28e6a371c6334c5b1a0d8656121905c

This should open a web page with the list of all available notebooks.  You can start with `Index.ipynb`,
or click on any notebook to launch.

To close the notebook and exit the container, you can press Ctrl-C. If you want to run the container in
the background and not take up your terminal, simply omit the `-it` options to `docker run` command.

See http://jupyter.org/ for more documentation on Jupyter.

### Connecting with a second bash shell

If you have the notebook server running and also want another window open with a bash shell, in another shell on your laptop you can do:

    $ docker exec -it riemann_book_container bash

### Updating the riemann_book files

In case the `riemann_book` repository changed since you built the docker image, you could do:

    root@...# cd $HOME
    root@...# git pull
    
### Updating `clawpack/riemann`

You may need some Riemann solvers not in the most recent release of Clawpack.  These can be obtained by checking out the master branch (and pulling any changes since you built the image, if necessary):

    root@...# cd $HOME/clawpack-5.4.0/riemann
    root@...# git checkout master
    root@...# git pull

If this brings down new Riemann solvers, you will need to compile them and re-install clawpack:

    root@...# cd $HOME/clawpack-5.4.0
    root@...# pip2 install -e .
    
### Update to the latest version of the notebooks

    root@...# cd $HOME
    root@...# git checkout master
    root@...# git pull
    
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
