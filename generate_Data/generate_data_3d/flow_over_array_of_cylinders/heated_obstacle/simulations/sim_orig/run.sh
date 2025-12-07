source /home/vboxuser/OpenFOAM/OpenFOAM-8/etc/bashrc 

decomposePar
mpirun -np 4 pisoFoam_write -parallel > pisoFoam_write.log
foamToVTK -useTimeName -noZero -ascii -noPointValues -fields '(delta_U delta_p U_non_cons p delta_U_prev delta_p_prev)' -excludePatches '(inlet outlet defaultFaces)' > log.foamtoVTK
