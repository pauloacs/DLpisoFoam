#!/bin/bash

# source OpenFOAM
source /opt/openfoam7/etc/bashrc
bash create_mesh.sh

source /opt/openfoam8/etc/bashrc
rm 0/cellLevel 0/C* 0/nSurfaceLayers  0/pointLevel  0/thickness*
renumberMesh -overwrite
decomposePar
mpirun  -np 4 DLpisoFoam -parallel
