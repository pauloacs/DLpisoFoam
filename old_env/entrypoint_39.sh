#!/bin/bash

#set -x

# This activates the conda environment
source /opt/conda/bin/activate
conda activate python39
. /home/repo/prep_env39.sh

# This compiles the solver
source /opt/openfoam8/etc/bashrc 
cd /home/repo/source
#
echo "Installing DLpisoFoam - algorithm 1 version" 
cd DLpisoFoam_alg1
wclean
wmake
cd ..
#
echo "Installing DLpisoFoam - algorithm 2 version"
cd DLpisoFoam_alg2
wclean
wmake
cd /home/repo

# Run extra commands
exec "$@"
