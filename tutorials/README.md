To get started with the ML-enhanced CFD solvers, run any of the available tutorials.

**DLbuoyantPimpleFoam** and **DLpisoFoam** require a pretrained ML surrogate model. A pre-trained model is provided so you can use the solver and observe how it accelerates convergence compared to classic CFD solvers.

**DLbuoyantPimpleFoam_auto** is self-contained. Simply run it, and an ML surrogate model will be trained automatically during the simulation. The model is continuously retrained to maintain optimal accuracy. You can customize the retraining process in `system/MLSamplingDict`.