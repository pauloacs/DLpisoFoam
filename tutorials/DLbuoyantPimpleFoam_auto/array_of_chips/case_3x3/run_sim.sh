#!/bin/bash

# Generate mesh
mv 0 0.orig
./genMesh.sh

# Run auto-trained ML aided CFD solver
echo "RUNNING DLbuoyantPimpleFoam_auto ..."
echo "####################################"
DLbuoyantPimpleFoam_auto

