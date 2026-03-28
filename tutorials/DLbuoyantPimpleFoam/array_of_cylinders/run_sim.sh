#!/bin/bash

# Generate mesh
mv 0 0.orig
./genMesh.sh
mv 0.orig 0

# Run auto-trained ML aided CFD solver
echo "RUNNING DLbuoyantPimpleFoam ..."
echo "####################################"
DLbuoyantPimpleFoam

