FROM jupyter/scipy-notebook:8e15d329f1e9

USER root
# Install some useful tools + gfortran
RUN apt-get update \
 && apt-get install -yq --no-install-recommends \
    dialog \
    net-tools \
    nano \
    gfortran \
    && apt-get clean \
&& rm -rf /var/lib/apt/lists/*
RUN apt-get install -y tar git curl wget dialog net-tools build-essential

User jovyan
# Install JSAnimation
# (Not necessary if we install Clawpack)
RUN pip install -e git+https://github.com/jakevdp/JSAnimation.git#egg=JSAnimation

# Add book's files
COPY . .

# Install other things needed for notebooks:
RUN pip2 install --no-cache-dir -r $HOME/requirements.txt
RUN pip install jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable equation-numbering/main

# Install clawpack-v5.4.0:
RUN pip2 install --src=$HOME --user -e git+https://github.com/clawpack/clawpack.git@v5.4.0#egg=clawpack-v5.4.0
