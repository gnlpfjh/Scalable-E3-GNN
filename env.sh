#! /bin/bash
conda create -y -n segnn python=3.10 numpy scipy matplotlib
conda activate segnn
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y pyg pytorch-cluster pytorch-scatter pytorch-sparse -c pyg # pytorch_geometric
conda install -y e3nn pytorch-lightning numba -c conda-forge
python -m pip install test-tube nvidia-ml-py rebound libconf
python -m pip install python-dev-tools
# conda install -c conda-forge libconfig