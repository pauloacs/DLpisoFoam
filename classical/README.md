# Classical / Offline CFD Workflow

This folder groups the assets for the classic workflow: prepare a dataset,
train a surrogate separately, then run CFD with the trained artifacts.

## Workflow Assets

- `gen_datasets/`: CFD cases and scripts for generating 2D/3D training data.
- `train_SM/`: offline training and evaluation scripts.
- `tutorials/`: classic `DLpisoFoam` and `DLbuoyantPimpleFoam` cases.
- `packages/`: classic `pressure_SM` and optional Tucker delta-delta package sources.
- `source/`: OpenFOAM solver implementations.

The `pressure_SM` package remains the classic 2D/3D solver interface. The
optional `pressure_SM_delta_delta` package provides the older 3D delta-delta
training/inference stack and Tucker compression useful for some offline
workflows; it is not an AUTO dependency.

## Build

Run these from the repository root:

```bash
wmake classical/source/DLpisoFoam
wmake classical/source/DLpisoFoam_2d
wmake classical/source/DLbuoyantPimpleFoam
```

See the [tutorial index](../tutorials/README.md) for examples and the
[repository README](../README.md) for environment setup.