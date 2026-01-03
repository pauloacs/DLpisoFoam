#/bin/bash
blockMesh -dict system/blockMeshDict_plate
if [ ! -d "constant/triSurface" ]; then
    mkdir -p constant/triSurface
fi
surfaceMeshTriangulate constant/triSurface/geometry.stl
surfaceFeatures
rm -r constant/polyMesh
# Run this instead if you want a coarse mesh
#blockMesh -dict system/blockMeshDict_block_coarse
blockMesh -dict system/blockMeshDict_block
snappyHexMesh -overwrite
checkMesh
