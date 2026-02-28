#!/bin/bash
# vim: set fileencoding=utf-8 fileformat=unix :
# -*- coding: utf-8 -*-
# vim: set ts=8 et sw=4 sts=4 sta :


rm -rf constant/polyMesh
blockMesh -dict "system/blockMeshDict" -region "."
#refineMesh -overwrite

# topoSet -dict "system/topoSetDict" -region "."
# refineHexMesh -overwrite -region "." -minSet highRes2

renumberMesh -overwrite -region "."

./repeatMesh.sh -no-blockMesh -x 3 -y 3

# ---------------------------------------------------------------------------
# Refine only the middle region based on the current mesh bounding box
# ---------------------------------------------------------------------------
echo "Capturing bounding box for middle-region refinement..."
bboxLine=$(checkMesh -region "${rg}" 2>/dev/null | awk '/Overall domain bounding box/ {print $5,$6,$7,$8,$9,$10}')

read xmin ymin zmin xmax ymax zmax <<<"$(echo "${bboxLine}" | tr -d '()')"

read xLo xHi yLo yHi zLo zHi <<< "$(python3 - "$xmin" "$ymin" "$zmin" "$xmax" "$ymax" "$zmax" <<'PY'
import sys
xmin, ymin, zmin, xmax, ymax, zmax = map(float, sys.argv[1:])
buffer = 0.0
def bounds(lo, hi):
    span = hi - lo
    return lo - buffer*span, hi + buffer*span
print(*bounds(xmin, xmax + 0.05), *bounds(ymin, ymax), *bounds(zmin, zmax - 0.005))
PY
)"

    cat > system/topoSetDict_mid <<EOF
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      topoSetDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

actions
(
    {
        name    middleRegion;
        type    cellSet;
        action  new;
        source  boxToCell;
        sourceInfo
        {
            box (${xLo} ${yLo} ${zLo}) (${xHi} ${yHi} ${zHi});
        }
    }
);
EOF

#########################################3

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

#refineWallLayer -overwrite "(top)" .5
#refineWallLayer -overwrite "(bot)" .5
#refineWallLayer -overwrite "(back)" .5
#refineWallLayer -overwrite "(front)" .5

#refineWallLayer -overwrite "(top)" .5
#refineWallLayer -overwrite "(bot)" .5
#refineWallLayer -overwrite "(back)" .5
#refineWallLayer -overwrite "(front)" .5

#refineWallLayer -overwrite "(chip)" .5
#refineWallLayer -overwrite "(chip)" .5

topoSet -dict "system/topoSetDict_mid" -region "."
#refineHexMesh -overwrite -region "." -minSet middleRegion


rm constant/polyMesh/refinementHistory

echo "Finished extrudeMesh."

refineMesh -overwrite

checkMesh -region "${rg}"

