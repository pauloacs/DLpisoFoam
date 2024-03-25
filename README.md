# DLpisoFoam


## Contents

- [Introduction](#introduction)
- [Running with Docker](#running-with-docker)

## Introduction

This repository contains the solvers and test cases for the **DLpisoFoam** solver. This solver is based on the OpenFOAM v8 version and is developed to solve the incompressible Navier-Stokes equations using the PISO algorithm. The main goal of this solver is to enhance the pressure-velocity coupling with a Deep Learning surrogate model. The solver is currently being developed in two versions:

1 - **DLpisoFoam-alg1 and DLpisoFoam-alg2**:
 the solvers developed in https://github.com/pauloacs/Solving-Poisson-s-Equation-through-DL-for-CFD-apllications. These solvers use the **U_to_p surrogate model** in surrogate_models/u_to_p/.

2 - **DLpisoFoam-deltas**:
  this solver is currently implemented, but the **deltau_to_delta surrogate models** are yet being developed in https://github.com/pauloacs/Solving-Poisson-Equation-with-DL-pt2. An example of the surrogate model can be found at surrogate_models/deltau_to_deltap/. 


Here you can find the DLpisoFoam solvers as well as test cases where those can be used. A DockerFile and everything that is needed to build the docker image is also provided here to ease the installation of the solver.

  ### Running with Docker

  To ensure reproducibility, a Docker container with the solver is provided. If you already have Docker installed, you can build your own docker image locally by running:

 ```sh
  $ docker build dlpisofoam .
  ```
  
  or pull the container with the following command:
  
  ```sh
  $ docker pull pauloacs/dlpisofoam:latest
  ```
  
  Using 
  
  ```sh
  $ docker run -it -v $(pwd):/home/repo --rm <image_name> bash
  ```
  This will create a Docker container and launch a shell.
