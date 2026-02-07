#!/bin/bash -l

#set -x

source /opt/openfoam8/etc/bashrc

# This activates the conda environment
#source /opt/conda/bin/activate
source /opt/conda/etc/profile.d/conda.sh
conda activate python311_solver
source /home/repo/prep_env311.sh

# This compiles the solver
source /opt/openfoam8/etc/bashrc 
cd /home/repo/source
#

echo ""
echo "################################################################################"
echo "################## Installing CFD SOLVERS ###################################"
echo "################################################################################"
echo ""

echo "Installing DLpisoFoam" 
cd DLpisoFoam
wclean
wmake
cd ..
#
echo "Installing DLbuoyantPimpleFoam"
cd DLbuoyantPimpleFoam
wclean
wmake

echo ""
echo "################################################################################"
echo "################## CFD SOLVERS installed ###################################"
echo "################################################################################"
echo ""


cd /home/repo
#

echo ""
echo "################################################################################"
echo "################## Installing Pressure Surrogate Model########################"
echo "################################################################################"
echo ""


echo "Installing surrogate_model Python modules"
pip install .
#

echo ""
echo "################################################################################"
echo "################## Everything ready! ########################################"
echo "################################################################################"
echo ""

# Run extra commands
exec "$@"
