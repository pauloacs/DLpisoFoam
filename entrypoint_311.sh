#!/bin/bash

#set -x

source /opt/openfoam8/etc/bashrc

# This activates the conda environment
source /opt/conda/bin/activate
conda activate python311_solver
. /home/repo/prep_env311.sh

# This compiles the solver
source /opt/openfoam8/etc/bashrc 
cd /home/repo/source
#
echo "Installing DLpisoFoam" 
cd DLpisoFoam
wclean
wmake
cd ..
#
echo "Installing DLbouyantPimpleFoam"
cd DLbouyantPimpleFoam
wclean
wmake

cd /home/repo
#
echo "Installing surrogate_model Python modules"
pip install .
#
# Run extra commands
exec "$@"
