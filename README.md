# DLpisoFoam

This repository contains the solvers developed in the **Solving Poisson's Equation through DL for CFD applications** Master's thesis.

The dissertation can be found at FEUP repository: https://sigarra.up.pt/feup/en/pub_geral.pub_view?pi_pub_base_id=547360

Here you can find the DLpisoFoam solvers and a test case where those can be used. A DockerFile and everything that is needed to build the docker image is also provided here to ease the installation of the solver.

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
  $ docker run -it --rm <image_name> bash
  ```
  This will create a Docker container and launch a shell. Inside the /home/foam directory, you'll find the solvers and a test case.
