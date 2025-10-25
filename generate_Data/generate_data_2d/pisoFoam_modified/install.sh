#!/bin/bash
#SBATCH --time=400:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -p batch    # partition(s)
#SBATCH --mem-per-cpu=1000M   # memory per CPU core
#SBATCH -J "install"   # job name
#SBATCH --mail-user=up201605045@fe.up.pt   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

module load openfoam/v1812
source $FOAM
module load mpi/openmpi-x86_64
module load gcc/7.2.1
wmake
