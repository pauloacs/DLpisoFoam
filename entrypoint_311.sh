#!/bin/bash -l

#set -x

#source /opt/openfoam8/etc/bashrc

# OpenFOAM interprets positional arguments to its bashrc as configuration
# options, so preserve the requested container command while sourcing it.
requested_command=("$@")
set --
source /opt/openfoam8/etc/bashrc
export FOAM_USER_APPBIN="$FOAM_APPBIN"
set -- "${requested_command[@]}"
unset requested_command

# Activate the solver Python environment for embedded Python.
source /opt/conda/etc/profile.d/conda.sh
conda activate python311_solver
source /home/repo/prep_env311.sh


# Build the classic and AUTO solvers from their workflow groups.
export PATH=/opt/conda/envs/python311_solver/bin:$PATH
echo ""
echo "################################################################################"
echo "################## Installing CFD SOLVERS ###################################"
echo "################################################################################"
echo ""

echo "Installing DLpisoFoam"
cd /home/repo/classical/source/DLpisoFoam
wclean
wmake

echo "Installing DLpisoFoam_2d"
cd /home/repo/classical/source/DLpisoFoam_2d
wclean
wmake
#
echo "Installing DLbuoyantPimpleFoam"
cd /home/repo/classical/source/DLbuoyantPimpleFoam
wclean
wmake

echo "Installing DLpimpleFoam_auto"
cd /home/repo/auto/source/DLpimpleFoam_auto
wclean
wmake

echo "Installing DLpimpleFoam_oracle"
cd /home/repo/auto/source/DLpimpleFoam_oracle
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
/opt/conda/envs/python311_solver/bin/python -m pip install '.[mesh_graph]'
#

echo ""
echo "################################################################################"
echo "################## Everything ready! ########################################"
echo "################################################################################"
echo ""

# Run extra commands
exec "$@"
