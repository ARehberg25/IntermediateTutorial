#!/usr/bin/env bash

module load python
virtualenv --no-download $HOME/jupyter_py3
source $HOME/jupyter_py3/bin/activate
mkdir -p ~/.local/share/jupyter/kernels
pip install --no-index ipykernel
python -m ipykernel install --user --name hk_ca --display-name "HKCA Python 3.x Kernel"
pip install --no-index numpy numpy h5py torch matplotlib pandas
