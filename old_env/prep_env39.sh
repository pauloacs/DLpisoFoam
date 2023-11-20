#!/bin/bash

export PYTHON_LIB_PATH=/opt/conda/envs/python39/lib
export PYTHON_BIN_PATH=/opt/conda/envs/python39/bin/python3.9/bin
export PYTHON_INCLUDE_PATH=/opt/conda/envs/python39/include/python3.9
export NUMPY_INCLUDE_PATH=/opt/conda/envs/python39/lib/python3.9/site-packages/numpy/core/include
export PYTHON_LIB_NAME=lpython3.9

export LD_LIBRARY_PATH=$PYTHON_LIB_PATH:$LD_LIBRARY_PATH
export LIBRARY_PATH=$PYTHON_LIB_PATH:$LIBRARY_PATH
export PATH=$PYTHON_BIN_PATH:$PATH
