# DLpisoFoam

**Deep Learning-Enhanced CFD Solvers for Accelerated Fluid Flow Simulations**

## 📋 Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [Publications](#publications)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Part 1: Environment Setup](#part-1-environment-setup)
    - [Option 1: Docker (Recommended)](#option-1-docker-recommended)
    - [Option 2: Local Installation](#option-2-local-installation)
  - [Part 2: Training your Surrogate Model](#part-2-training-your-surrogate-model)
  - [Part 3: Running Your First CFD using the Surrogate Model](#part-3--running-your-first-cfd-using-the-surrogate-model)
- [Available Solvers](#available-solvers)
  - [DLpisoFoam](#dlpisofoam)
  - [DLbuoyantPimpleFoam](#dlbuoyantpimplefoam)
- [Legacy Solvers](#legacy-solvers)
- [Citation](#citation)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)


## 🎯 Introduction

This repository contains **machine learning-enhanced OpenFOAM solvers** that accelerate computational fluid dynamics (CFD) simulations by replacing the computationally expensive pressure Poisson equation solver with a deep learning surrogate model.

**DLpisoFoam** and **DLbuoyantPimpleFoam** are based on OpenFOAM v8 and implement the PISO/PIMPLE algorithms with integrated neural network surrogate models for improved pressure-velocity coupling.

This is an **all-in-one repository** containing both surrogate models and CFD solvers, building upon the work from [Solving-Poisson's-Equation-through-DL-for-CFD-applications](https://github.com/pauloacs/Solving-Poisson-s-Equation-through-DL-for-CFD-apllications).

---

## ✨ Key Features

- 🚀 **Accelerated simulations**: Up to 20% less iterations required to convergence when compared to standard OpenFOAM solvers
- 🧠 **ML-powered**: Machine Learning surrogate models enhance pressure Poisson solver
- 🌡️ **Multiple physics**: Supports isothermal and thermal flows
- 🔧 **OpenFOAM compatible**: Drop-in replacement for `pisoFoam` and `buoyantPimpleFoam`
- 🐳 **Docker ready**: Pre-built containers for easy deployment
- 📊 **2D & 3D**: Surrogate models available for both 2D and 3D simulation cases

---

## 📚 Publications

### 1. Surrogate Model Development
**Application of machine learning to model the pressure Poisson equation for fluid flow on generic geometries**  
[Springer - Neural Computing and Applications (2024)](https://link.springer.com/article/10.1007/s00521-024-09935-0)
- Detailed description of the ML surrogate model architecture and training

### 2. DLpisoFoam Benchmark
**Enhancing CFD solver with Machine Learning techniques**  
[Computers & Fluids (2024)](https://www.sciencedirect.com/science/article/pii/S004578252400389X)
- Performance benchmarks and validation of DLpisoFoam solver

### 3. Improved Pressure-Velocity Coupling
**Surrogate-Based Pressure-Velocity Coupling: Accelerating Incompressible CFD Flow Solvers with Machine Learning**  
[ResearchGate (2024)](https://www.researchgate.net/publication/394002762_Surrogate-Based_Pressure-Velocity_Coupling_Accelerating_Incompressible_Cfd_Flow_Solvers_with_Machine_Learning)
- Enhanced pressure surrogate model and integration methodology

### 4. 3D Extensions *(Coming Soon)*
- Extension to 3D surrogate models and complex geometries

---

## 📁 Repository Structure

```
DLpisoFoam/
├── source/                      # Solver source code
│   ├── DLpisoFoam/             # Incompressible isothermal solver
│   └── DLbuoyantPimpleFoam/    # Thermal flow solver
├── pressure_SM/                 # Surrogate models
│   ├── 2D/                     # 2D models
│   │   ├── train_and_eval/    # Training & evaluation scripts
│   │   └── CFD_usable/        # Production-ready models
│   └── 3D/                     # 3D models
│       ├── train_and_eval/
│       └── CFD_usable/
├── CFD_test_case/              # Example test cases
├── other_solvers/              # Legacy solver versions
├── Dockerfile                  # Docker build configuration
└── env_311.yml                 # Python environment specification
```

---


## 🚀 Getting Started

This guide will walk you through:
1. Setting up your environment (Docker or local)
2. Training a surrogate model (SM) from scratch
3. Running CFD simulations with the trained model

---

### Prerequisites

- **For Docker**: Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- **For local install**: 
  - OpenFOAM v8
  - Conda or Miniconda
  - Python 3.11

---

## Part 1: Environment Setup

### Option 1: Docker (Recommended)

**Easiest method with automated setup and guaranteed reproducibility.**

#### Pull pre-built image:
```bash
docker pull pauloacs/dlpisofoam:latest
```

#### Or build locally:
```bash
docker build -t dlpisofoam .
```

#### Run container:
```bash
docker run -it -v $(pwd):/home/repo --rm pauloacs/dlpisofoam bash
```

This mounts your current directory inside the container at `/home/repo`.

---

### Option 2: Local Installation

#### Step 1: Create Python environment
```bash
conda env create -f env_311.yml
conda activate python311_solver
```

#### Step 2: Install surrogate model packages
```bash
python -m pip install -e .
```

#### Step 3: Set up OpenFOAM environment
```bash
source /opt/openfoam8/etc/bashrc  # Adjust path to your OpenFOAM installation
./prep_env311.sh                   # Set Python/NumPy paths
```

**Note**: You may need to modify `prep_env311.sh` with paths to your conda environment:
```bash
export PYTHON_INCLUDE_PATH=$CONDA_PREFIX/include/python3.11
export NUMPY_INCLUDE_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/numpy/core/include
export PYTHON_LIB_PATH=$CONDA_PREFIX/lib
export PYTHON_LIB_NAME=lpython3.11
```

#### Step 4: Compile the solvers

**For DLpisoFoam:**
```bash
cd source/DLpisoFoam
wclean
wmake
```

**For DLbuoyantPimpleFoam:**
```bash
cd source/DLbuoyantPimpleFoam
wclean
wmake
```

Verify installation:
```bash
which DLpisoFoam
DLpisoFoam -help
```

---

## Part 2: Training your Surrogate Model

(TO BE WRITTEN)

## Part 3: 🏃 Running Your First CFD using the Surrogate Model

### Example: DLpisoFoam Test Case

```bash
# Navigate to test case
cd CFD_test_case/DLpisoFoam/

# Run mesh generation (if needed)
blockMesh

# Run the solver
DLpisoFoam
```

### Post-processing

```bash
# Open in ParaView
paraFoam

# Or use built-in post-processing
postProcess -func 'mag(U)'
```

---

## 🔧 Available Solvers

### DLpisoFoam
**For incompressible, isothermal flows**

- Based on OpenFOAM's `pisoFoam`
- Uses pressure surrogate model to accelerate PISO algorithm
- Ideal for: laminar/turbulent flows, external aerodynamics, internal flows

**Example applications:**
- Flow over cylinders
- Channel flows

---

### DLbuoyantPimpleFoam
**For thermal flows with buoyancy**

- Based on OpenFOAM's `buoyantPimpleFoam`
- Handles natural/mixed convection
- Supports both incompressible and weakly-compressible flows

**Example applications:**
- Electronics cooling
- HVAC simulations

---

## 🏛️ Legacy Solvers

Previous versions are maintained in `other_solvers/`:

- **DLpisoFoam-alg1**: Original algorithm from initial publication
- **DLpisoFoam-alg2**: Intermediate version with U→p surrogate model

These use the older `u_to_p` surrogate models and are kept for reproducibility of earlier papers.

---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@article{dlpisofoam2024,
  title = {Enhancing CFD solver with Machine Learning techniques},
  author = {Sousa, Paulo and Rodrigues, Carlos Veiga and Afonso, Alexandre},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  volume = {429},
  pages = {117133},
  year = {2024},
  issn = {0045-7825},
  doi = {10.1016/j.cma.2024.117133},
  url = {https://www.sciencedirect.com/science/article/pii/S004578252400389X},
  keywords = {CFD, Machine Learning, Incompressible flows, PISO, OpenFOAM}
}
```

---

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/[your-username]/DLpisoFoam/issues)
- **Email**: [pauloacunhasousa@hotmail.com]
- **ResearchGate**: (https://www.researchgate.net/profile/Paulo-Sousa-74?ev=hdr_xprf)

---

## Acknowledgments

This work builds upon:
- [OpenFOAM Foundation](https://openfoam.org/)
- [Original Poisson Solver Repository](https://github.com/pauloacs/Solving-Poisson-s-Equation-through-DL-for-CFD-apllications)

---