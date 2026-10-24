The repository contains two surrogate workflows plus an oracle diagnostic.
This file is the top-level tutorial index; workflow assets live in
`classical/tutorials/` and `auto/tutorials/`.

- **Classic/offline**: `DLpisoFoam`, `DLpisoFoam_2d`, and
	`DLbuoyantPimpleFoam` load a surrogate trained before the production run.
- **AUTO**: `DLpimpleFoam_auto` samples CFD fields, trains an initial model,
	and activates it during the same simulation. Configure it in
	`system/MLSamplingDict` and the case's `python_module.py`.
- **Oracle**: `DLpimpleFoam_oracle` replays reference pressure/velocity fields
	to measure integration potential. It does not train or load a surrogate.

The solver sources and packages are grouped in `classical/` and `auto/` to
match these roles. A thermal AUTO solver is not tracked in this checkout.

## Compact PIMPLE Tutorials

The five compact AUTO/oracle cases use the tested PIMPLE integration on a
small 512-cell channel with a square obstacle. They contain only text
inputs: no generated mesh, restart snapshot, dataset, trained weights, or logs.
Each `Allrun` builds and checks its own mesh.
Classic solver examples are linked separately in the table below.

| Case | Prediction / Oracle Input | Integration |
| --- | --- | --- |
| [AUTO gridded pressure-only](../auto/tutorials/DLpimpleFoam_auto/gridded_shift/pressure_only) | Lattice CNN shifter `ddp` residual; fresh automatic training | Fixed-pressure momentum reconstruction |
| [AUTO graph pressure-only](../auto/tutorials/DLpimpleFoam_auto/graph_shift/pressure_only) | Mesh-graph shifter `ddp` residual; fresh automatic training | Fixed-pressure momentum reconstruction |
| [AUTO graph joint](../auto/tutorials/DLpimpleFoam_auto/graph_shift/joint) | Mesh-graph shifter `ddp` and `ddU` residuals; fresh automatic training | Joint state injection, pressure force, then reconstruction |
| [Oracle pressure-only](../auto/tutorials/DLpimpleFoam_oracle/pressure_only) | Perfect reference pressure, without interpolation | `rerunUEqnWithOraclePressure` |
| [Oracle joint](../auto/tutorials/DLpimpleFoam_oracle/joint) | Perfect reference pressure and velocity increments, without interpolation | `pressureForceThenReconstruct` |
| [Classic DLpisoFoam](../classical/tutorials/DLpisoFoam) | Offline-trained surrogate examples | PISO integration |
| [Classic buoyant solver](../classical/tutorials/DLbuoyantPimpleFoam) | Offline-trained surrogate examples | Buoyant PIMPLE integration |

### Prerequisites

Follow the repository setup instructions, source the OpenFOAM 8 environment,
and build the current solver sources. Older installed binaries may not support
the pressure-only oracle-compatible mode.

```bash
wmake auto/source/DLpimpleFoam_auto
wmake auto/source/DLpimpleFoam_oracle
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
matplotlib, and tensorly. Its package is `pressure_SM_gridded_shift`;
the graph framework uses `pressure_SM_mesh_graph`. The gridded example uses
`cnn_shifter_lightweight` on one full-domain `17 x 17 x 17` lattice block,
with no Tucker decomposition. Joint velocity prediction is currently
demonstrated only by the graph framework; the lattice shifter does not support
the graph model's total-velocity/residual-velocity head.

Graph models can also enable `include_dDivHbyA_input`, `include_dHbyA_input`,
and `add_divRAUGradDpPrev_input` for additional CFD-computed pressure-source
information. Together these add five channels. They are computed before
coarsening and normalized by the velocity scale, consistently in training and
inference. Changing these flags requires fresh selected-field samples and
retraining; existing weights with a different feature manifest are rejected.

Graph training defaults to `graph_loss='coarse_mse'`. An optional native-cell
objective can be enabled in the case Python module:

```python
graph_loss = 'native_obstacle_gradient'
loss_obstacle_patch = 'obstacle'
loss_obstacle_fraction = 0.5
loss_obstacle_radius = 0.05
loss_pressure_gradient_weight = 0.2
loss_velocity_gradient_weight = 0.1
```

To retain the coarse objective while adding a moderate native-cell signal, use
the blended objective instead:

```python
graph_loss = 'coarse_native_blend'
graph_native_loss_fraction = 0.25
```

The blend fraction interpolates the complete coarse and native objectives:
`0.0` is coarse-only and `1.0` is native-only. The default `0.25` keeps most
of the coarse loss while adding native value and gradient supervision. It is
independent of `loss_obstacle_fraction`, which controls the native objective's
global-versus-obstacle weighting.

The value terms compare pressure increments and total velocity increments
after the same limited prolongation, baseline composition, and output weighting
used at inference. The known timestep-start fields cancel from these errors.
Pressure errors are volume-mean-free; velocity errors are not mean-subtracted.
The native-cell objective mixes globally volume-weighted error with a smooth
obstacle-focused distribution. `loss_obstacle_fraction` is this mixture's
coefficient, not the exact weight inside the radius. Distances use the nearest
sampled obstacle-face centre, not an exact surface SDF. Gradient terms penalize
native internal-face owner/neighbour directional differences; their scales are
fit only on training frames. They are not OpenFOAM finite-volume momentum or
continuity residuals, and do not constrain `ddU` to be divergence-free.

This objective differentiates the Python coarse-to-native transfer, not the
OpenFOAM reconstruction/PIMPLE solves or boundary correction. It requires
sampled obstacle geometry and fine-cell targets, uses additional memory, and
does not change the model's spatial resolution. Retraining is required.
Compare physical native metrics, rather than the numerical magnitude of this
loss against the old coarse loss. Training exports `loss_components.csv`,
`native_metrics.csv`, and `native_summary.json` inside its training-plots folder,
separately from the coarse validation summary.

Existing samples can be reused read-only for a controlled loss ablation:

```bash
TRAIN_SCRIPT_MODE=1 python3 -m pressure_SM_mesh_graph._3D.training \
	--data_dir ML_data --data_file /path/to/completed/ML_data/data.h5
```

Run from the new case directory so its mesh and Python configuration are used;
model outputs go only to `--data_dir`. Add `--epochs 2` for an integration pilot,
not an accuracy comparison. No solver recompilation is needed for this loss.

For graph models, `graph_linear_prolongation=True` and
`velocity_linear_prolongation=True` enable slope-limited linear reconstruction
from coarse nodes to CFD cells. The limiter preserves each aggregate's
volume-weighted mean and prevents new extrema outside that node and its
face-connected coarse neighbours. `False` keeps piecewise-constant injection.
Velocity residual reconstruction still adds the native `dUPrev` baseline
afterward. Limiting cannot fix erroneous NN outputs or enforce continuity;
normal CFD correction remains necessary. Existing models need no retraining
to use this reconstruction change; restart the solver to load the updated code.

Linear transfer also makes three weak interface adjustments using only native
internal faces between aggregates. It penalizes deviations from the coarse
least-squares gradient rather than raw field differences, preserving aggregate
volume means and local bounds at every step. The NumPy inference and
differentiable native-loss transfers use the same adjustment. It cannot repair
an incorrect coarse prediction or bound the subsequently added history baseline.
For a diagnostic comparison with the previous limited reconstruction, call
`prolong(..., smooth=False)`; `linear=False` still selects constant injection.

Validation retains the coarse-graph plots and additionally writes
`training_plots/<target>/full_resolution/`. Every held-out sample has pressure
and joint-velocity error distributions comparing the graph with the full CFD
mesh, with linear prolongation both on and off. Separate `prolongation_true/`
and `prolongation_false/` folders contain full-resolution statistics; pressure
maps are limited to the first three samples. Histograms and metrics use all
cells with volume weights, in physical units, after inference-equivalent
baseline composition and before OpenFOAM boundary correction/reconstruction.
`prediction_metrics.csv` is written even when `plot_training=False`.

To regenerate diagnostics without retraining or changing checkpoint artifacts,
run from a case with a configuration matching the saved checkpoint:

```bash
TRAIN_SCRIPT_MODE=1 python3 -m pressure_SM_mesh_graph._3D.training \
	--data_dir ML_data --evaluate_only --plot_dir ML_data/diagnostics_recomputed
```

Enable `plot_training=True` for figures. After online retraining, use the same
`--window_frames` as that update to reproduce its chronological split. Do not
evaluate a dataset while it is being written by a solver or trainer.

Pressure-only models can optionally initialize reconstruction with a temporal
velocity trend instead of the unadvanced timestep-start velocity. In
`system/MLSamplingDict`, set:

```cpp
smUseVelocityExtrapolation true;
smVelocityExtrapolationAlpha 1.0;
```

This applies `U = U_start + alpha * (deltaT / previousDeltaT) * dUPrev` once,
then corrects velocity boundary conditions and rebuilds `phi` before the selected
pressure-force/reconstruction steps. `dUPrev` is the preceding completed
timestep's converged velocity minus its start velocity; its associated timestep
is cached explicitly for adaptive stepping. The switch defaults to `false`.
Zero alpha also preserves the disabled initialization, including the original
`phi`. No history means no extrapolated increment. Use this only with
`smUseInputPredictorState=false` and no NN velocity output; incompatible
joint-output or LTS configurations are rejected. It can initialize either
post-injection reconstruction or the direct handoff described below.
No NN retraining is needed. Rebuild the solver after updating its C++ source,
and set the controls before launching a new run.

### Zero Reconstruction

To inject predictions and immediately start normal PIMPLE with a fresh momentum
matrix, use the following in `system/MLSamplingDict`:

```cpp
smUseInputPredictorState false;
postSMInjectionReconstruction false;
smStateReconstructionCorrectors 0;
```

This mode restores timestep-start `U/phi`, injects predicted pressure and the
optional total velocity increment, clears feature-stage residuals, and starts
outer corrector 1. It does not reuse the feature-generation momentum matrix.
The pressure-force stage is disabled when post-injection reconstruction is off,
even if `smPressureForceThenReconstruct=true` remains in the dictionary. Pressure
patch conditions are refreshed from the fresh momentum matrix before its solve.

To keep the no-solve pressure-force update but skip fixed-pressure momentum
reconstruction solves, use:

```cpp
postSMInjectionReconstruction true;
smPressureForceThenReconstruct true;
smStateReconstructionCorrectors 0;
```

The minimum-corrector setting is ignored when no reconstruction correctors are
requested. Both modes support oracle-compatible stopping controls. They require
the current solver build, but no NN retraining. Fewer reconstruction solves can
increase subsequent PIMPLE work; compare total timestep time at the same
convergence tolerances rather than assuming zero correctors is faster.

### Run

Choose a case directory and run its script, for example:

```bash
cd auto/tutorials/DLpimpleFoam_auto/graph_shift/pressure_only
bash Allrun
```

For the gridded shifter, use `auto/tutorials/DLpimpleFoam_auto/gridded_shift/pressure_only`.

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
supported but disabled by the sampling settings in these examples; see the
root README's Case Controls section to enable it. The gridded tutorial also
schedules only initial training.
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