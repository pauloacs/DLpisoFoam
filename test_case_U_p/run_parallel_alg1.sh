#!/bin/bash
decomposePar

# The number of processors can be changed in system/decomposeParDict
mpirun -np 2 DLpisoFoam_alg1 -parallel