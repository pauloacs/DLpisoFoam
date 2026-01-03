## How to run the solver 

To run just select a test case and run:

```sh
  $ DLpisoFoam_alg1
```
or
```sh
  $ DLpisoFoam_alg2
```

inside.

To run with multiple processor run the 'run-parallel' scripts:

```sh
  $ . run_parallel_alg1.sh
```
or
```sh
  $ . run_parallel_alg2.sh
```

## Debugging the surrogate model python code

Run 

```sh
  § pip install pytest
  $ python -m pytest surrogate_model/
```

for debugging the surrogate model code without running OF. 

!!! Trying to debug the python code while running 'DLpisoFoam' is not advised !!!