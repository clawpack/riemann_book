
# Install anaconda Python stack and some other useful tools
FROM continuumio/anaconda
RUN apt-get update
RUN apt-get install -y tar git curl wget dialog net-tools build-essential

# Install editors:
RUN apt-get install -y vim nano

# Install JSAnimation
# (Not necessary if we install Clawpack)
RUN pip install -e git+https://github.com/jakevdp/JSAnimation.git#egg=JSAnimation

# Install gfortran
RUN apt-get install -y gfortran

# Get riemann_book files:
RUN git clone https://github.com/clawpack/riemann_book

# Install other things needed for notebooks:
RUN pip install -r riemann_book/requirements.txt
RUN jupyter nbextension enable --py widgetsnbextension
RUN pip install jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable equation-numbering/main

# Install JupyterHub, required for this to work on mybinder
RUN pip install --no-cache-dir jupyterhub==0.7.2

# Install clawpack-v5.4.0:
RUN pip install --src=/ --user -e git+https://github.com/clawpack/clawpack.git@v5.4.0#egg=clawpack-v5.4.0

