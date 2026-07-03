#ifndef SurrogateModel_H
#define SurrogateModel_H

#include "fvCFD.H"
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL POD_ARRAY_API2

// Only the implementation file (SurrogateModel.C) should import the array API.
// All other translation units that include this header get NO_IMPORT_ARRAY.
#ifndef SURROGATE_MODEL_IMPL
#define NO_IMPORT_ARRAY
#endif

#include <numpy/arrayobject.h>


class SurrogateModel
{
    // --- References to mesh and fields ---
    const Foam::fvMesh& mesh_;
    Foam::volScalarField& p_;
    Foam::volVectorField& U_;
    Foam::volVectorField& dUStar_;
    Foam::volVectorField& dUStarPrev_;
    Foam::volScalarField& dpML_;
    Foam::volScalarField& dp_;
    Foam::volScalarField& dpPrev_;
    Foam::volScalarField& p_prev_;
    // Fields for delta_delta
    Foam::volVectorField& ddUStar_;
    Foam::volVectorField& ddUStarPrev_;
    Foam::volScalarField& ddpML_;
    Foam::volScalarField& ddpPrev_;
    Foam::volScalarField& ddp_;
    Foam::volVectorField& grad_dpPrev_;
    Foam::volScalarField& laplace_dpPrev_;
    Foam::volScalarField& uDotGradDpPrev_;
    Foam::volScalarField& gradDpPrevMag_;
    Foam::volVectorField& ddUStarML_;  // ML-predicted non-conservative velocity correction
    // Divergence of predicted velocity for ML input
    Foam::volScalarField& divUFirstPred_;
    // Pressure-correction velocity fields (previous steps)
    Foam::volVectorField& dUCorrPrev_;   // previous dUCorr (input feature)
    Foam::volVectorField& ddUCorrPrev_;  // previous ddUCorr (input feature)
    // PISO pressure-equation quantities (input features)
    Foam::volScalarField& rAU_ML_;       // 1/A(U)
    Foam::volVectorField& HbyA_ML_;      // rAU*H(U)
    Foam::volScalarField& divHbyA_ML_;   // div(HbyA)
    Foam::volVectorField& dHbyA_ML_;     // HbyA - HbyA_prev (temporal variation)
    Foam::volScalarField& dDivHbyA_ML_;  // divHbyA - divHbyA_prev (temporal variation)
    Foam::volVectorField& rAUGradDpPrev_ML_;    // rAU * grad(dpPrev)
    Foam::volScalarField& divRAUGradDpPrev_ML_; // div(rAU*grad(dpPrev))
    Foam::volScalarField& pressureEqResidualp_ML_; // div(HbyA) - div(rAU*grad(dpPrev))
    Foam::volVectorField& rAUGradpPrev_ML_;      // rAU * grad(p_prev)
    Foam::volScalarField& divRAUGradpPrev_ML_;   // div(rAU*grad(p_prev))

    // --- Python / NumPy state ---
    double U_MAX_NORM_;
    double (*input_vals_init_)[11];
    double (*input_vals_)[46];  // previous 37 + rAUGradDpPrev(3) + divRAUGradDpPrev(1) + pressureEqResidualp(1) + rAUGradpPrev(3) + divRAUGradpPrev(1)
    double (*input_vals_z_top_)[3];
    double (*input_vals_z_bot_)[3];
    double (*input_vals_y_top_)[3];
    double (*input_vals_y_bot_)[3];
    double (*input_vals_obst_)[3];

    PyObject* py_func_;
    PyObject* py_args_;
    PyObject* init_func_;
    PyObject* init_args_;
    PyObject* reload_func_;
    PyObject* array_3d_;
    PyObject* array_3d_z_top_;
    PyObject* array_3d_z_bot_;
    PyObject* array_3d_y_top_;
    PyObject* array_3d_y_bot_;
    PyObject* array_3d_obst_;
    PyObject* array_3d_init_;

    npy_intp dim_np_[2];

    bool initialized_;

public:
    SurrogateModel
    (
        const Foam::fvMesh& mesh,
        Foam::volScalarField& p,
        Foam::volVectorField& U,
        Foam::volVectorField& dUStar,
        Foam::volVectorField& dUStarPrev,
        Foam::volScalarField& dpML,
        Foam::volScalarField& dp,
        Foam::volScalarField& dpPrev,
        Foam::volScalarField& p_prev,
        Foam::volVectorField& ddUStar,
        Foam::volVectorField& ddUStarPrev,
        Foam::volScalarField& ddpML,
        Foam::volScalarField& ddpPrev,
        Foam::volScalarField& ddp,
        Foam::volVectorField& grad_dpPrev,
        Foam::volScalarField& laplace_dpPrev,
        Foam::volScalarField& uDotGradDpPrev,
        Foam::volScalarField& gradDpPrevMag,
        Foam::volVectorField& ddUStarML,
        Foam::volScalarField& divUFirstPred,
        Foam::volVectorField& dUCorrPrev,
        Foam::volVectorField& ddUCorrPrev,
        Foam::volScalarField& rAU_ML,
        Foam::volVectorField& HbyA_ML,
        Foam::volScalarField& divHbyA_ML,
        Foam::volVectorField& dHbyA_ML,
        Foam::volScalarField& dDivHbyA_ML,
        Foam::volVectorField& rAUGradDpPrev_ML,
        Foam::volScalarField& divRAUGradDpPrev_ML,
        Foam::volScalarField& pressureEqResidualp_ML,
        Foam::volVectorField& rAUGradpPrev_ML,
        Foam::volScalarField& divRAUGradpPrev_ML
    );

    ~SurrogateModel();

    void init();
    void predict();
    void reload();
    bool initialized() const { return initialized_; }
};

#endif
