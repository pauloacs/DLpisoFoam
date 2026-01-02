#!/bin/bash
# vim: set fileencoding=utf-8 fileformat=unix :
# -*- coding: utf-8 -*-
# vim: set ts=8 et sw=4 sts=4 sta :
########################################################################
#
# Name: repeatMesh.sh
# Changelog (version number, date and author):
#   * 1.0, 2018-01-17, Carlos Veiga Rodrigues <cvrodrigues@gmail.com>
# 
# Copyright and license: GPLv3
# Copyright (C) 2018 by Carlos Veiga Rodrigues. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# For more details consult the GNU General Public License at:
# <http://www.gnu.org/licenses/gpl.html>.
#
########################################################################


################################
##
## Supporting functions
##
################################
_usage () {
    echo ""
    echo "Usage: ${0} [OPTIONS]"
    echo "Description:"
    echo "  Repeats a blockMesh grid to have x * y * z instances of the"
    echo "  original grid. The number of instances (x, y, z) in one direction"
    echo "  is equal to the number of repetitions in that direction minus one."
    echo "Options:"
    echo "  -x <integer>      number of instances in X direction"
    echo "  -y <integer>      number of instances in Y direction"
    echo "  -z <integer>      number of instances in Z direction"
    echo "  -no-blockMesh     do not run blockMesh"
    #echo "  -case <dir>       specify alternate case directory, default is the cwd"
    echo "  -dict <file>      specify alternative dictionary for the blockMesh description"
    echo "  -region <name>    specify alternative mesh region"
    echo "  -help             print the usage"
    echo ""
}

_errPrintUsage () {
    echo "Error: ${@}" ; _usage ; exit 1
}

_errCheck () {
    if [ ! $? -eq 0 ] ; then
        echo "Error: ${@}"
        exit 1
    fi
}

_isInt () {
    [[ ! $@ =~ ^[-+]?[0-9]+$ ]] && _errPrintUsage "argument '${@}' not an integer"
}

_checkArg () {
    for opt in "x" "y" "z" "dict" "region" "help" ; do
        if [[ "${2}" =~ ^("${opt}"|"-${opt}"|"--${opt}") ]] ; then
            _errPrintUsage "option ${1} with invalid argument '${2}'"
        fi
    done
}

_removeAuxPath () {
    DIR="${@}"
    if [ -d "${DIR}" ] ; then
        read -p "Path '${DIR}' will be deleted, proceed? [y for yes] " yn
        case $yn in
            [Yy]*)
                rm -rI "${DIR}"
                [ ! $? -eq 0 ] && exit 1
                ;;
            *) 
                echo "Unable to proceed, exiting."
                exit 1
                ;;
        esac
    fi
}


## Default values
nx=1
ny=1
nz=1
dict="system/blockMeshDict"
rg="."
run_blockMesh=1

[ $# -eq 0 ] && _errPrintUsage "no options given, ${0} requires at least one option"

while [ $# -gt 0 ] ; do
    case "${1}" in
        "-x" | "--x" )
            [ -z $2 ] && _errPrintUsage "option '${1}' needs an argument"
            _checkArg $1 $2
            nx=$2
            shift
            ;;
        "-y" | "--y" )
            [ -z $2 ] && _errPrintUsage "option '${1}' needs an argument"
            _checkArg $1 $2
            ny=$2
            shift
            ;;
        "-z" | "--z" )
            [ -z $2 ] && _errPrintUsage "option '${1}' needs an argument"
            _checkArg $1 $2
            nz=$2
            shift
            ;;
        "-no-blockMesh" | "--no-blockMesh" )
            run_blockMesh=0
            ;;
        "-dict" | "--dict" )
            [ -z "${2}" ] && _errPrintUsage "option '${1}' needs an argument"
            _checkArg $1 $2
            dict="${2}"
            shift
            ;;
        "-region" | "--region" )
            [ -z "${2}" ] && _errPrintUsage "option '${1}' needs an argument"
            _checkArg $1 $2
            rg="${2}"
            shift
            ;;
        "-help" | "--help" ) _usage ; exit 0 ;;
        *) _errPrintUsage "argument '${1}' is not a valid option" ;;
    esac
    shift
done

_isInt "${nx}"
_isInt "${ny}"
_isInt "${nz}"
max=$nx
for n in "${ny}" "${nz}" ; do
    ((n > max)) && max=$n
done
if [ $max -le 1 ] ; then
    echo  "$(($nx - 1)) repeats in X" \
        ", $(($ny - 1)) repeats in Y" \
        ", $(($nz - 1)) repeats in Z"
    echo "... nothing to do, exiting"
    _usage
    exit 0
else
    echo  "Repeats in X: $(($nx - 1))"
    echo  "Repeats in Y: $(($ny - 1))"
    echo  "Repeats in Z: $(($nz - 1))"
fi


################################
##
## Prepare folder
##
################################
_removeAuxPath "constant/auxRegion"
_removeAuxPath "constant/auxRegionAll"


################################
##
## Generate base region
##
################################
if [ "${run_blockMesh}" == "1" ] ; then
    echo "Running blockMesh ..."
    _removeAuxPath "constant/${rg}/polyMesh"
    blockMesh -dict "${dict}" -region "${rg}"
    _errCheck "blockMesh failed"
else
    echo "Skipping blockMesh"
fi


################################
##
## Get dimensions
##
################################
bb=$(checkMesh -region "${rg}" | grep -i "domain *bounding *box")
if [ -z "${bb}" ] ; then
    echo "Error: could not find bounding box in output of checkMesh"
    echo "       Suggestion, run checkMesh"
    exit 1
fi
#bb=$(echo ${bb//[!0-9. ()]/})  # remove leading & trainling whitespace
bb=${bb//[!0-9. ()]/}
bb=${bb#"${bb%%[![:space:]]*}"}  # remove leading spaces
bb=${bb%"${bb##*[![:space:]]}"}  # remove trailing spaces

nlp=$((10#$(echo -n "${bb//[!\(]/}" | wc -m)))
nrp=$((10#$(echo -n "${bb//[!\)]/}" | wc -m)))
if [ "$nlp" -lt 2 ] || [ "$nrp" -lt 2 ] ; then
    echo "Error: unable to parse checkMesh bounding box output"
    echo "       expecting two arrays delimited with '(...)'"
    echo "       bounding box output: '${bb}'"
    exit 1
fi
xyz_i=$(echo -n $bb | cut -d "(" -f2 | cut -d ")" -f1)
xyz_f=$(echo -n $bb | cut -d "(" -f3 | cut -d ")" -f1)
#xyz_i=$(echo -n $bb | awk -F"[()]" '{print $2}')
#xyz_f=$(echo -n $bb | awk -F"[()]" '{print $4}')

xi=$(echo $xyz_i | cut -d" " -f1)
xf=$(echo $xyz_f | cut -d" " -f1)
xdim=$(echo "${xf} ${xi}" | awk '{print $1 - $2}')
yi=$(echo $xyz_i | cut -d" " -f2)
yf=$(echo $xyz_f | cut -d" " -f2)
ydim=$(echo "${yf} ${yi}" | awk '{print $1 - $2}')
zi=$(echo $xyz_i | cut -d" " -f3)
zf=$(echo $xyz_f | cut -d" " -f3)
zdim=$(echo "${zf} ${zi}" | awk '{print $1 - $2}')


################################
##
## Repeats in X
##
################################
if [ $nx -gt 1 ] ; then
    ## Create region to merge
    #blockMesh -dict system/blockMeshDict -region auxRegion
    mkdir -p constant/auxRegion
    cp -r constant/"${rg}"/polyMesh constant/auxRegion/
    
    ## Name stitching region
    sed -i 's/inlet/stitch_inlet/g' constant/auxRegion/polyMesh/boundary
    
    i=1
    while [ $i -lt $nx ] ; do
        if [ $((2 * $i)) -le $nx ] ; then
            mkdir -p constant/auxRegionAll
            cp -r constant/"${rg}"/polyMesh constant/auxRegionAll
            sed -i 's/inlet/stitch_inlet/g' \
                constant/auxRegionAll/polyMesh/boundary
            sed -i 's/outlet/stitch_outlet/g' constant/"${rg}"/polyMesh/boundary
            delta=$(echo "${i} ${xdim}" | awk '{print $1 * $2}')
            transformPoints -region auxRegionAll -translate "(${delta} 0 0)"
            _errCheck "transformPoints failed"
            mergeMeshes -overwrite -masterRegion "${rg}" \
                -addRegion auxRegionAll . .
            _errCheck "mergeMeshes failed"
            stitchMesh -overwrite -region "${rg}" \
                -perfect stitch_outlet stitch_inlet
            _errCheck "stitchMesh failed"
            for t in $(foamListTimes -withZero); do
                [ -f "${t}/meshPhi" ] && rm -f "${t}"/meshPhi
            done
            rm -rf constant/auxRegionAll
            ## translate auxRegion to have it at same position
            transformPoints -region auxRegion -translate "(${delta} 0 0)"
            _errCheck "transformPoints failed"
            i=$((2 * $i))
        else
            ## Repeat region
            sed -i 's/outlet/stitch_outlet/g' constant/"${rg}"/polyMesh/boundary
            transformPoints -region auxRegion -translate "(${xdim} 0 0)"
            _errCheck "transformPoints failed"
            mergeMeshes -overwrite -masterRegion "${rg}" \
                -addRegion auxRegion . .
            _errCheck "mergeMeshes failed"
            stitchMesh -overwrite -region "${rg}" \
                -perfect stitch_outlet stitch_inlet
            _errCheck "stitchMesh failed"
            for t in $(foamListTimes -withZero); do
                [ -f "${t}/meshPhi" ] && rm -f "${t}"/meshPhi
            done
            i=$(($i + 1))
        fi
        renumberMesh -overwrite -region "${rg}"
        _errCheck "renumberMesh failed"
        sed -i 's/stitch_outlet/emptyBoundary'${i}'e/g' \
            constant/"${rg}"/polyMesh/boundary
        sed -i 's/stitch_inlet/emptyBoundary'${i}'w/g' \
            constant/"${rg}"/polyMesh/boundary
        #checkMesh
    done
    rm -rf constant/auxRegion
fi


################################
##
## Repeats in Y using main grid
##
################################
if [ $ny -gt 1 ] ; then
    ## Create region to merge
    mkdir -p constant/auxRegion
    cp -r constant/"${rg}"/polyMesh constant/auxRegion/

    ## Name stitching region
    sed -i 's/front/stitch_front/g' constant/auxRegion/polyMesh/boundary

    j=1
    while [ $j -lt $ny ] ; do
        if [ $((2 * $j)) -le $ny ] ; then
            mkdir -p constant/auxRegionAll
            cp -r constant/"${rg}"/polyMesh constant/auxRegionAll
            sed -i 's/front/stitch_front/g' \
                constant/auxRegionAll/polyMesh/boundary
            sed -i 's/back/stitch_back/g' constant/"${rg}"/polyMesh/boundary
            delta=$(echo "${j} ${ydim}" | awk '{print $1 * $2}')
            transformPoints -region auxRegionAll -translate "(0 ${delta} 0)"
            _errCheck "transformPoints failed"
            mergeMeshes -overwrite -masterRegion "${rg}" \
                -addRegion auxRegionAll . .
            _errCheck "mergeMeshes failed"
            stitchMesh -overwrite -region "${rg}" \
                -perfect stitch_back stitch_front
            _errCheck "stitchMesh failed"
            for t in $(foamListTimes -withZero); do
                [ -f "${t}/meshPhi" ] && rm -f "${t}"/meshPhi
            done
            rm -rf constant/auxRegionAll
            ## translate auxRegion to have it at same position
            transformPoints -region auxRegion -translate "(0 ${delta} 0)"
            _errCheck "transformPoints failed"
            j=$((2 * $j))
        else
            ## Repeat region
            sed -i 's/back/stitch_back/g' constant/"${rg}"/polyMesh/boundary
            transformPoints -region auxRegion -translate "(0 ${ydim} 0)"
            _errCheck "transformPoints failed"
            mergeMeshes -overwrite -masterRegion "${rg}" \
                -addRegion auxRegion . .
            _errCheck "mergeMeshes failed"
            stitchMesh -overwrite -region "${rg}" \
                -perfect stitch_back stitch_front
            _errCheck "stitchMesh failed"
            for t in $(foamListTimes -withZero); do
                [ -f "${t}/meshPhi" ] && rm -f "${t}"/meshPhi
            done
            j=$(($j + 1))
        fi
        renumberMesh -overwrite -region "${rg}"
        _errCheck "renumberMesh failed"
        sed -i 's/stitch_back/emptyBoundary'${j}'n/g' \
            constant/"${rg}"/polyMesh/boundary
        sed -i 's/stitch_front/emptyBoundary'${j}'s/g' \
            constant/"${rg}"/polyMesh/boundary
        #checkMesh
    done
    rm -rf constant/auxRegion
fi


################################
##
## Repeats in Z using main grid
##
################################
if [ $nz -gt 1 ] ; then
    ## Create region to merge
    mkdir -p constant/auxRegion
    cp -r constant/"${rg}"/polyMesh constant/auxRegion/

    ## Name stitching region
    sed -i 's/bot/stitch_bot/g' constant/auxRegion/polyMesh/boundary

    k=1
    while [ $k -lt $nz ] ; do
        if [ $((2 * $k)) -le $nz ] ; then
            mkdir -p constant/auxRegionAll
            cp -r constant/"${rg}"/polyMesh constant/auxRegionAll
            sed -i 's/bot/stitch_bot/g' \
                constant/auxRegionAll/polyMesh/boundary
            sed -i 's/top/stitch_top/g' constant/"${rg}"/polyMesh/boundary
            delta=$(echo "${k} ${zdim}" | awk '{print $1 * $2}')
            transformPoints -region auxRegionAll -translate "(0 0 ${delta})"
            _errCheck "transformPoints failed"
            mergeMeshes -overwrite -masterRegion "${rg}" \
                -addRegion auxRegionAll . .
            _errCheck "mergeMeshes failed"
            stitchMesh -overwrite -region "${rg}" \
                -perfect stitch_top stitch_bot
            _errCheck "stitchMesh failed"
            for t in $(foamListTimes -withZero); do
                [ -f "${t}/meshPhi" ] && rm -f "${t}"/meshPhi
            done
            rm -rf constant/auxRegionAll
            ## translate auxRegion to have it at same position
            transformPoints -region auxRegion -translate "(0 0 ${delta})"
            _errCheck "transformPoints failed"
            k=$((2 * $k))
        else
            ## Repeat region
            sed -i 's/top/stitch_top/g' constant/"${rg}"/polyMesh/boundary
            transformPoints -region auxRegion -translate "(0 0 ${zdim})"
            _errCheck "transformPoints failed"
            mergeMeshes -overwrite -masterRegion "${rg}" \
                -addRegion auxRegion . .
            _errCheck "mergeMeshes failed"
            stitchMesh -overwrite -region "${rg}" \
                -perfect stitch_top stitch_bot
            _errCheck "stitchMesh failed"
            for t in $(foamListTimes -withZero); do
                [ -f "${t}/meshPhi" ] && rm -f "${t}"/meshPhi
            done
            k=$(($k + 1))
        fi
        renumberMesh -overwrite -region "${rg}"
        _errCheck "renumberMesh failed"
        sed -i 's/stitch_top/emptyBoundary'${k}'t/g' \
            constant/"${rg}"/polyMesh/boundary
        sed -i 's/stitch_bot/emptyBoundary'${k}'b/g' \
            constant/"${rg}"/polyMesh/boundary
        #checkMesh
    done
    rm -rf constant/auxRegion
fi

## clear empty patches
## BUG #1, pyFoamClearEmptyBoundaries.py fails sometimes
## BUG #2, pyFoamClearEmptyBoundaries.py does not support regions
# command -v pyFoamClearEmptyBoundaries.py
# if [ $? -eq 0 ] ; then
#     pyFoamClearEmptyBoundaries.py .
#     _errCheck "pyFoamClearEmptyBoundaries.py failed"
#     renumberMesh -overwrite -region "${rg}"
#     _errCheck "renumberMesh failed"
# fi

## clear empty patches... apparently this works
attachMesh -overwrite

checkMesh -region "${rg}"

#paraFoam -region "${rg}"
#paraFoam -builtin -region "${rg}"