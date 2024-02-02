# How to run the solver 

To run just select a test case and run:

''DLpisoFoam_alg1''
or
''DLpisoFoam_agl2'' 

inside.

To run with multiple processor run the 'run-parallel' scripts:

'' . run_parallel_alg1.sh"
or
'' . run_parallel_alg2.sh"

# Debugging

Run 

''python test_module.py'

for debugging the surrogate model code without running OF. 

!!! Trying to debug the python code while running 'DLpisoFoam' is not advised !!!