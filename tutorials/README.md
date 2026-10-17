To get started with the ML-enhanced CFD solvers, run any of the available tutorials.

**DLbuoyantPimpleFoam** and **DLpisoFoam** require a pretrained ML surrogate model. A pre-trained model is provided so you can use the solver and observe how it accelerates convergence compared to classic CFD solvers.

**DLbuoyantPimpleFoam_auto** is self-contained. Simply run it, and an ML surrogate model will be trained automatically during the simulation. The model is continuously retrained to maintain optimal accuracy. You can customize the retraining process in `system/MLSamplingDict`.

## Compact PIMPLE Tutorials

These five serial OpenFOAM Foundation 8 cases use the tested PIMPLE integration
on a small 512-cell channel with a square obstacle. They contain only text
inputs: no generated mesh, restart snapshot, dataset, trained weights, or logs.
Each `Allrun` builds and checks its own mesh.

| Case | Prediction / Oracle Input | Integration |
| --- | --- | --- |
| [DLpimpleFoam_auto/gridded_shift/pressure_only](DLpimpleFoam_auto/gridded_shift/pressure_only) | Lattice CNN shifter `ddp` residual; fresh automatic training | Fixed-pressure momentum reconstruction |
| [DLpimpleFoam_auto/graph_shift/pressure_only](DLpimpleFoam_auto/graph_shift/pressure_only) | Mesh-graph shifter `ddp` residual; fresh automatic training | Fixed-pressure momentum reconstruction |
| [DLpimpleFoam_auto/graph_shift/joint](DLpimpleFoam_auto/graph_shift/joint) | Mesh-graph shifter `ddp` and `ddU` residuals; fresh automatic training | Joint state injection, pressure force, then reconstruction |
| [DLpimpleFoam_oracle/pressure_only](DLpimpleFoam_oracle/pressure_only) | Perfect reference pressure, without interpolation | `rerunUEqnWithOraclePressure` |
| [DLpimpleFoam_oracle/joint](DLpimpleFoam_oracle/joint) | Perfect reference pressure and velocity increments, without interpolation | `pressureForceThenReconstruct` |

### Prerequisites

Follow the repository setup instructions, source the OpenFOAM 8 environment,
and build the current solver sources. Older installed binaries may not support
the pressure-only oracle-compatible mode.

```bash
wmake source/DLpimpleFoam_auto
wmake source/DLpimpleFoam_oracle
```

For either surrogate framework, activate the project's Python 3.11 solver environment
(TensorFlow 2.14 / NumPy 1.x) with `python3` on `PATH` and its libraries on
`LD_LIBRARY_PATH`. Install the package and serial-MPI Python interface from the
repository root:

```bash
python3 -m pip install -e '.[mesh_graph]' mpi4py
```

The gridded framework also uses the scientific Python dependencies in the
project's solver environment, including pandas, SciPy, scikit-learn, numba,
matplotlib, and tensorly. Its package is `pressure_SM_delta_delta_shift`;
the graph framework uses `pressure_SM_mesh_graph`. The gridded example uses
`cnn_shifter_lightweight` on one full-domain `17 x 17 x 17` lattice block,
with no Tucker decomposition. Joint velocity prediction is currently
demonstrated only by the graph framework; the lattice shifter does not support
the graph model's total-velocity/residual-velocity head.

### Run

Choose a case directory and run its script, for example:

```bash
cd tutorials/DLpimpleFoam_auto/graph_shift/pressure_only
bash Allrun
```

For the gridded shifter, use `tutorials/DLpimpleFoam_auto/gridded_shift/pressure_only`.

The default executables are `DLpimpleFoam_auto` and `DLpimpleFoam_oracle`.
The current `DLpimpleFoam_auto` supports both pressure-only and joint
pressure/velocity prediction; no alternate solver name is required.
Set `SOLVER` only to select a differently named or separately installed build.

Run these examples serially. Graph inference currently supports a single CFD
rank. Inspect `log.blockMesh`, `log.checkMesh`, and
`log.solver` after the run. Graph training writes its baseline summary under
`ML_data/training_plots/ddp/`; set `plot_training=True` in the case Python module
to generate figures. Sampling and reconstruction are controlled by
`system/MLSamplingDict`; oracle cases instead use `system/oracleDict`.
The gridded trainer writes `ML_data/weights.h5`, normalization files, loss plots,
and prediction plots inside `ML_data/`.

The defaults run 12 timesteps. Both frameworks collect six training samples;
graph training uses five epochs, while gridded training uses eight so that the
existing lattice trainer passes its checkpoint warm-up and saves weights.
These are integration demonstrations, not mesh-independent physical results or
speedup benchmarks. Increase resolution, sample diversity, and training time
for meaningful accuracy or timing studies. Graph incremental retraining is
disabled in these examples; the gridded tutorial also schedules only initial training.
The oracle executes both a reference and a replay
pass, so it supplies an idealized diagnostic baseline, not a deployable model.

### Clean

`Allrun` refuses to overwrite an existing mesh or solver log. To start fresh:

```bash
bash Allclean
```

This deletes generated meshes, nonzero time directories, logs, datasets,
trained models, and inspection output in that case, while retaining `0/` and
all case inputs. Generated outputs are ignored by Git; do not force-add them.