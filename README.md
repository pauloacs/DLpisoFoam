# DLpisoFoam

## Contents

- [Introduction](#introduction)
- [Running with Docker](#running-with-docker)
- [Local Setup](#local-setup)
- [Run your first tutorial](#run-your-first-tutorial)

# Articles

1 - Application of machine learning to model the pressure Poisson equation for fluid flow on generic geometries:
https://link.springer.com/article/10.1007/s00521-024-09935-0
  - This paper contains a detailed description of the ML Surrogate Model developed to solve the Pressure Poisson Equation.

2- Enhancing CFD solver with Machine Learning techniques: 
https://www.sciencedirect.com/science/article/pii/S004578252400389X
  - This paper contains the DLpisoFoam CFD solver benchmark

3- Hybrid Cfd - a Data-Driven Approach to Speed-Up Incompressible Cfd Solvers:
[https://www.researchgate.net/publication/389522957_Hybrid_Cfd_-_a_Data-Driven_Approach_to_Speed-Up_Incompressible_Cfd_Solvers](https://www.researchgate.net/publication/394002762_Surrogate-Based_Pressure-Velocity_Coupling_Accelerating_Incompressible_Cfd_Flow_Solvers_with_Machine_Learning)
  - This paper presents an improved pressure SM and its integration in the DLpisoFoam CFD solver

## Introduction

This repository contains the solvers and test cases for the **DLpisoFoam** and **DLbuoyantPimpleFoam** solvers.
These solvers are based on the OpenFOAM v8 version and are developed to solve the Navier-Stokes equations using the PISO/PIMPLE algorithm. The primary objective of this solver is to improve the pressure-velocity coupling using a Deep Learning surrogate model.

This **all-in-one repository (Surrogate models + DL aided CFD Solvers)** builds upon the work from https://github.com/pauloacs/Solving-Poisson-s-Equation-through-DL-for-CFD-apllications.

### Surrogate Models

Stored under pressure_SM/ and available in 2D or 3D versions. To be used to:

1 - Train and evaluate the SM: pressure_SM/_3D/train_and_eval
2 - Integrate the SM into the CFD solver: pressure_SM/_3D/CFD_usable


### Latest solvers:

These make usage of the above surrogate models to accelerate OpenFOAM fluid flow solvers.

**DLpisoFoam**:
  This Deep-Learning aided CFD solver makes usage of the developed pressure surrogate models for accelerating **incompressible isothermal fluid flows**.

**DLbuoyantPimpleFoam**:
  This Deep-Learning aided CFD solver makes usage of the developed pressure surrogate models for accelerating **thermal fluid flows**.
  

## How to start

### Setup the enviroment and install the CFD solvers

There are two ways to use DLpisoFoam:

#### 1. **Running with Docker**:

This is the easiest option as the setup of the environment is automated. To ensure reproducibility, a Docker container with the solver is provided. If you already have Docker installed, you can build your own docker image locally by running:

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

#### 2. **Local Setup**: This method requires manual setup and installation.

To set up the environment locally, follow these steps:

1. Create a Python conda virtual environment by running the following command:

```sh
$ conda env create -f env_311.yml
```

2. Install the surrogate model Python packages:

```sh
$ python -m pip install .
```

3. Make sure the required environment variables for the CFD solver are properly set by running:

```sh
$ ./prep_env311.sh
```

Note: You may need to create your own `prep_env311.sh` file with the correct path to your conda environment.

4. Finally, install the CFD solvers. For example, to install DLpisoFoam, navigate to the `source/DLpisoFoam` directory and run the following commands:

```sh
$ wclean
$ wmake
```

### Run your first tutorial

To run your first tutorial, navigate to the directory of the solver you want to test. For example, if you want to run DLpisoFoam, follow these steps:

1. Change to the `CFD_test_case/` directory:

```sh
$ cd test_case_deltaU_deltaP/
```

2. Run the DLpisoFoam solver:

```sh
$ cd DLpisoFoam/
$ DLpisoFoam
```


### Old versions - stored under **other_solvers/**

**DLpisoFoam-alg1 and DLpisoFoam-alg2**:
 the solvers developed in https://github.com/pauloacs/Solving-Poisson-s-Equation-through-DL-for-CFD-apllications. These solvers use the **U_to_p surrogate model** in surrogate_models/u_to_p/. 

Here, you can find the DLpisoFoam solvers, as well as test cases that utilize them. A DockerFile and everything that is needed to build the docker image is also provided here to ease the installation of the solver.