source /home/vboxuser/OpenFOAM/OpenFOAM-8/etc/bashrc 

decomposePar
mpirun -np 4 pisoFoam_write -parallel > pisoFoam_write.log
