#!/bin/bash

# This script sets up the environment for the Python 3.11 build for the docker image
# You need to adjust the paths to your own installation paths if not using docker

export PYTHON_LIB_PATH=/opt/conda/envs/python311/lib
export PYTHON_BIN_PATH=/opt/conda/envs/python311/bin/python3.11/bin
export PYTHON_INCLUDE_PATH=/opt/conda/envs/python311/include/python3.11
export NUMPY_INCLUDE_PATH=/opt/conda/envs/python311/lib/python3.11/site-packages/numpy/core/include
export PYTHON_LIB_NAME=lpython3.11

export LD_LIBRARY_PATH=$PYTHON_LIB_PATH:$LD_LIBRARY_PATH
export LIBRARY_PATH=$PYTHON_LIB_PATH:$LIBRARY_PATH
export PATH=$PYTHON_BIN_PATH:$PATH
