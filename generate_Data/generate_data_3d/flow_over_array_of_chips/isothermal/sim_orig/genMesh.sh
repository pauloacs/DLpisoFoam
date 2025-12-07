#!/bin/bash
# vim: set fileencoding=utf-8 fileformat=unix :
# -*- coding: utf-8 -*-
# vim: set ts=8 et sw=4 sts=4 sta :


rm -rf constant/polyMesh
blockMesh -dict "system/blockMeshDict" -region "."
#refineMesh -overwrite

topoSet -dict "system/topoSetDict" -region "."
#refineHexMesh -overwrite -region "." -minSet highRes1
refineHexMesh -overwrite -region "." -minSet highRes2

#refineMesh -overwrite -dict "system/refineMeshDict"

#refineWallLayer -overwrite "(cylinder)" .5

renumberMesh -overwrite -region "."

./repeatMesh.sh -no-blockMesh -x 4 -y 3


################################
##
## Add inlet and outlet buffers
##
################################
echo "Adding inlet and outlet buffers..."

echo "Running extrudeMesh..."

cp system/extrudeMeshDict_front system/extrudeMeshDict
extrudeMesh -region "${rg}" # > extrudeMesh_inlet.log 2>&1

cp system/extrudeMeshDict_back system/extrudeMeshDict
extrudeMesh -region "${rg}" # > extrudeMesh_inlet.log 2>&1

cp system/extrudeMeshDict_inlet system/extrudeMeshDict
extrudeMesh -region "${rg}" # > extrudeMesh_inlet.log 2>&1

cp system/extrudeMeshDict_outlet system/extrudeMeshDict
extrudeMesh -region "${rg}" #> extrudeMesh_outlet.log 2>&1

refineWallLayer -overwrite "(top)" .5
refineWallLayer -overwrite "(bot)" .5
refineWallLayer -overwrite "(back)" .5
refineWallLayer -overwrite "(front)" .5

#refineWallLayer -overwrite "(top)" .5
#refineWallLayer -overwrite "(bot)" .5
#refineWallLayer -overwrite "(back)" .5
#refineWallLayer -overwrite "(front)" .5

refineWallLayer -overwrite "(chips)" .5

rm constant/polyMesh/refinementHistory

echo "Finished extrudeMesh."
checkMesh -region "${rg}"

