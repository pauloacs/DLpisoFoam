#!/bin/bash

#set -x

# This activates the conda environment
source /opt/conda/bin/activate
conda activate python39
. /home/repo/prep_env39.sh

# This compiles the solver
source /opt/openfoam8/etc/bashrc 
cd /home/repo/DLPoissonSolver
wclean
wmake

# Run extra commands
exec "$@"
