# DLpisoFoam

**Deep Learning-Enhanced CFD Solvers for Accelerated Fluid Flow Simulations**

## 📋 Contents

- [Introduction](#-introduction)
- [What This Repository Enables](#-what-this-repository-enables)
- [Key Features](#-key-features)
- [Publications](#-publications)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Part 1: Environment Setup](#part-1-environment-setup)
    - [Option 1: Docker (Recommended)](#option-1-docker-recommended)
    - [Option 2: Local Installation](#option-2-local-installation)
  - [Part 2: Preparing Datasets & Training Your Surrogate Model](#part-2-preparing-datasets--training-your-surrogate-model)
    - [Part 2.1: Generate CFD Training Data](#part-21-generate-cfd-training-data)
    - [Part 2.2: Train the Surrogate Model](#part-22-train-the-surrogate-model)
  - [Part 3: Running Your First CFD Simulation with the Surrogate Model](#part-3--running-your-first-cfd-simulation-with-the-surrogate-model)
    - [Part 3.1: Run the DLpisoFoam Test Case with the PRE-TRAINED Surrogate Model](#part-31-run-the-dlpisofoam-test-case-with-the-pre-trained-surrogate-model)
    - [Part 3.2: Run DLpisoFoam with YOUR trained Surrogate model](#part-32-run-dlpisofoam-with-your-trained-surrogate-model)
- [Available CFD Solvers](#-available-cfd-solvers)
  - [DLpisoFoam](#dlpisofoam)
  - [DLbuoyantPimpleFoam](#dlbuoyantpimplefoam)
- [Legacy Solvers](#-legacy-solvers)
- [Citation](#-citation)
- [Contact](#-contact)
- [Acknowledgments](#acknowledgments)


## 🎯 Introduction

This repository contains **machine learning-enhanced OpenFOAM solvers** that accelerate computational fluid dynamics (CFD) simulations by replacing the computationally expensive pressure Poisson equation solver with a deep learning surrogate model.

**DLpisoFoam** and **DLbuoyantPimpleFoam** are based on OpenFOAM v8 and implement the PISO/PIMPLE algorithms with integrated neural network surrogate models for improved pressure-velocity coupling.

This is an **all-in-one repository** containing both surrogate models and CFD solvers, building upon the work from [Solving-Poisson's-Equation-through-DL-for-CFD-applications](https://github.com/pauloacs/Solving-Poisson-s-Equation-through-DL-for-CFD-apllications).

---

## 🎯 What This Repository Enables

This repository provides a complete workflow for accelerating CFD simulations with machine learning:
1. **Generate Training Data**: Run CFD simulations to create comprehensive datasets from various flow scenarios (examples in `gen_datasets/`)
2. **Train Surrogate Models**: Develop neural network models that learn to predict pressure corrections in unsteady CFD simulations (examples in `train_SM/`)
3. **Deploy in Production**: Integrate trained models with custom OpenFOAM-based solvers for accelerated simulations (examples in `tutorials/`)

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
[Neural Computing and Applications (2024)](https://link.springer.com/article/10.1007/s00521-024-09935-0)
- Detailed description of the ML surrogate model architecture and training

### 2. DLpisoFoam Benchmark
**Enhancing CFD solver with Machine Learning techniques**  
[Computer Methods in Applied Mechanics and Engineering (2024)](https://www.sciencedirect.com/science/article/pii/S004578252400389X)
- Performance benchmarks and validation of DLpisoFoam solver

### 3. Improved Pressure-Velocity Coupling
[SSRN Preprint (2025)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5364744)
- Enhanced pressure surrogate model and integration methodology

### 4. 3D Extensions *(Coming Soon)*
- Extension to 3D surrogate models and complex geometries

---

## 📁 Repository Structure

```
DLpisoFoam/
├── source/                  # Solver source code
│   ├── DLpisoFoam/           # Incompressible isothermal solver
│   └── DLbuoyantPimpleFoam/  # Thermal flow solver
├── pressure_SM/             # Surrogate models
│   ├── 2D/                   # 2D models
│   │   ├── train_and_eval/    # Training & evaluation scripts
│   │   └── CFD_usable/        # Interface for integration with CFD solver
│   └── 3D/                   # 3D models
│       ├── train_and_eval/    # Training & evaluation scripts
│   │   └── CFD_usable/        # Interface for integration with CFD solver
├── tutorials/                # Example test cases
├── gen_datasets/             # Dataset generation scripts
├── other_solvers/            # Legacy solver versions
├── Dockerfile                # Docker build configuration
└── env_311.yml               # Python environment specification
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

Verify installations:
```bash
which DLpisoFoam
DLpisoFoam -help
```

```bash
which DLbuoyantPimpleFoam
DLbuoyantPimpleFoam -help
```

---

## Part 2: Preparing Datasets & Training Your Surrogate Model

### Part 2.1: Generate CFD Training Data

Use the scripts in `gen_datasets/` to generate training datasets from various flow scenarios:

- **Confined flows over chip arrays**: Electronics cooling simulations
- **Flow around squared cylinders**: Bluff body aerodynamics
- **Flow over inclined plates**: Boundary layer studies

All configurations support both **isothermal** and **thermal** conditions to match your application requirements.

Modified versions of `pisoFoam` and `buoyantPimpleFoam` are included to extract the necessary flow field data for training. Depending on the simulation type, generation can be fully or semi-automatic.

---

### Part 2.2: Train the Surrogate Model

Navigate to `train_SM/` and use the `run_train.sh` scripts to train your model:

```bash
cd train_SM/
# Edit run_train.sh to set your datasetPath
./run_train.sh
```

**Key Configuration:**
- **Required**: Set `datasetPath` to your training dataset location
- **Optional**: Adjust hyperparameters (learning rate, batch size, dropout, etc.)

**Training Outputs:**
- **Model file**: HDF5 format TensorFlow model (e.g., `model_MLP_small-std-drop0.1-lr0.0005-regNone-batch1024.h5`)
- **Compression artifacts**: Tucker factors (3D cases) or PCA vectors (2D cases) (required for CFD deployment)

These files are ready to use with the CFD solvers - the model weights/biases load automatically via TensorFlow.

---

## Part 3: 🏃 Running Your First CFD Simulation with the Surrogate Model

### Part 3.1: Run the DLpisoFoam Test Case with the PRE-TRAINED Surrogate Model

```bash
# Navigate to test case
cd tutorials/DLpisoFoam/

# Run mesh generation (if needed)
./genMesh.sh

# Run the solver
DLpisoFoam
```

```bash
# Open in ParaView
paraFoam

# Or use built-in post-processing
postProcess -func 'mag(U)'
```

---

### Part 3.2: Run DLpisoFoam with YOUR trained Surrogate model

1. **Copy the example test case:**
  ```bash
  cp -r tutorials/DLpisoFoam/array_of_cyks tutorials/DLpisoFoam/my_custom_test
  cd tutorials/DLpisoFoam/my_custom_test
  ```

2. **Copy your trained surrogate model files:**
  ```bash
  # Copy your TensorFlow model
  cp /path/to/your/model.h5 .
  
  # Copy compression artifacts (Tucker factors (3D cases) or PCA vectors (2D cases))
  cp /path/to/your/tucker_factors.pkl .
  
  # Copy normalization files
  cp /path/to/maxs .
  cp /path/to/mean_std.npz .
  ```

---


## 🔧 Available CFD Solvers

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

If you use this work in your research, you can cite the papers that introduced it:

```bibtex
@article{firstPressureSM2024,
  author = {Sousa, Paulo and Afonso, Alexandre and Veiga Rodrigues, Carlos},
  year = {2024},
  month = {05},
  pages = {1-26},
  title = {Application of machine learning to model the pressure poisson equation for fluid flow on generic geometries},
  volume = {36},
  journal = {Neural Computing and Applications},
  doi = {10.1007/s00521-024-09935-0}
}
```

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

```bibtex
@unpublished{surrogateBasedPressureVelocity2024,
  title = {Surrogate-Based Pressure–Velocity Coupling: Accelerating Incompressible CFD Flow Solvers with Machine Learning},
  author = {Sousa, Paulo Araújo da Cunha and Afonso, Alexandre M. and Veiga Rodrigues, Carlos},
  year = {2024},
  note = {Preprint},
  doi = {10.2139/ssrn.5364744},
  url = {https://ssrn.com/abstract=5364744}
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