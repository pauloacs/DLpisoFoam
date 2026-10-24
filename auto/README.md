# AUTO CFD Workflow

This folder groups the current AUTO and oracle tutorial cases. The AUTO
solver samples fields, trains an initial surrogate, and activates it during
the same simulation. The oracle replays reference fields to assess integration
potential; it does not train or load a surrogate.

## Current Components

- `tutorials/DLpimpleFoam_auto/`: gridded pressure-only and graph pressure-only/joint cases.
- `tutorials/DLpimpleFoam_oracle/`: pressure-only and joint reference-injection cases.
- `source/DLpimpleFoam_auto/`: self-training OpenFOAM solver.
- `source/DLpimpleFoam_oracle/`: oracle diagnostic solver.
- `packages/`: `pressure_SM_gridded_shift` and `pressure_SM_mesh_graph` source;

## Build and Run

Run builds from the repository root:

```bash
wmake auto/source/DLpimpleFoam_auto
wmake auto/source/DLpimpleFoam_oracle
cd auto/tutorials/DLpimpleFoam_auto/graph_shift/pressure_only
bash Allrun
```

The AUTO case's `python_module.py` selects the model and features;
`system/MLSamplingDict` controls sampling, training, and activation. Start with
the [tutorial index](../tutorials/README.md) and [repository setup guide](../README.md).