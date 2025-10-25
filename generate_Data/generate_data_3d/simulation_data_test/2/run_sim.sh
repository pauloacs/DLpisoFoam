#!/bin/bash
#SBATCH --time=100:00:00   # walltime
#SBATCH --ntasks=4  # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -p big,batch    # partition(s)
#SBATCH --mem-per-cpu=1000M   # memory per CPU core
#SBATCH -J "plate0"   # job name
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# source OpenFOAM
source /opt/openfoam7/etc/bashrc
#bash create_mesh.sh

rm 0/cellLevel 0/C* 0/nSurfaceLayers  0/pointLevel  0/thickness*
renumberMesh -overwrite
decomposePar
mpirun  -np 4 pisoFoam_modied -parallel
reconstructPar -newTimes
postProcess -func writeCellCentres

foamToVTK -useTimeName -noZero -ascii -noPointValues -fields '(delta_U delta_p U_non_cons p Cx Cy Cz delta_U_prev delta_p_prev)' -excludePatches '(inlet outlet defaultFaces)' > log.foamtoVTK
