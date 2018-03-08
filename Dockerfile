FROM ubuntu:16.04

# Install some useful tools + gfortran
RUN apt-get update \
 && apt-get install -yq \
    locales \
    dialog \
    net-tools \
    tar \
    git \
    nano \
    vim \
    curl \
    wget \
    gfortran \
    build-essential \
    python \
    python-dev \
    virtualenv \
    && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV VIRTUAL_ENV /srv/venv
ENV PATH ${VIRTUAL_ENV}/bin:${PATH}

# Use bash as default shell, rather than sh
ENV SHELL /bin/bash

# Set up user
ENV NB_USER jovyan
ENV NB_UID 1000
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

WORKDIR ${HOME}

RUN mkdir -p ${VIRTUAL_ENV} && chown ${NB_USER}:${NB_USER} ${VIRTUAL_ENV}

User jovyan

RUN virtualenv ${VIRTUAL_ENV}
ENV PYTHONHOME ${VIRTUAL_ENV}

# Install notebook extensions
RUN pip install --no-cache-dir \
    jupyter \
    jupyter_contrib_nbextensions \
    jupyterhub-legacy-py2-singleuser==0.7.2

RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable widgetsnbextension --py
RUN jupyter nbextension enable equation-numbering/main

# Install clawpack-v5.4.0:
RUN pip2 install --src=$HOME/clawpack -e git+https://github.com/clawpack/clawpack.git@v5.4.0#egg=clawpack-v5.4.0

# Add book's files
RUN git clone --depth=1 https://github.com/clawpack/riemann_book

#move into the riemann_book directory
WORKDIR ${HOME}/riemann_book

RUN pip install --no-cache-dir -r $HOME/riemann_book/requirements.txt

CMD jupyter notebook --ip='*' --no-browser
