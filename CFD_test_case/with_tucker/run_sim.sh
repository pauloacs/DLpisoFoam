#!/bin/bash

# source OpenFOAM
source /opt/openfoam7/etc/bashrc
bash create_mesh.sh

rm 0/cellLevel 0/C* 0/nSurfaceLayers  0/pointLevel  0/thickness*
renumberMesh -overwrite
decomposePar
mpirun  -np 4 pisoFoam_modied -parallel
reconstructPar -newTimes
postProcess -func writeCellCentres

foamToVTK -useTimeName -noZero -ascii -noPointValues -fields '(delta_U delta_p U_non_cons p Cx Cy Cz delta_U_prev delta_p_prev)' -excludePatches '(inlet outlet defaultFaces)' > log.foamtoVTK
