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
    Foam::volScalarField& p_rgh_;
    Foam::volScalarField& p_;
    Foam::volVectorField& U_;
    Foam::volScalarField& rho_;
    Foam::volVectorField& delta_U_;
    Foam::volVectorField& delta_U_prev_;
    Foam::volScalarField& delta_p_rgh_;
    Foam::volScalarField& delta_p_rgh_CFD_;
    Foam::volScalarField& delta_p_rgh_prev_;
    Foam::volScalarField& p_rgh_prev_;
    // New fields for delta_delta
    Foam::volVectorField& delta_delta_U_;
    Foam::volVectorField& delta_delta_U_prev_;
    Foam::volScalarField& delta_delta_p_rgh_;
    Foam::volScalarField& delta_delta_p_rgh_prev_;
    Foam::volScalarField& delta_delta_p_rgh_CFD_;

    // --- Python / NumPy state ---
    double U_MAX_NORM_;
    double (*input_vals_init_)[11];
    double (*input_vals_)[18];  // U(3) + delta_delta_U(3) + delta_delta_U_prev(3) + delta_delta_p_rgh_prev(1) + delta_p_rgh_prev(1) + delta_U(3) + p_rgh_prev(1) + div_U(1) + div_dU(1) + div_delta_delta_U(1)
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
        Foam::volScalarField& p_rgh,
        Foam::volScalarField& p,
        Foam::volVectorField& U,
        Foam::volScalarField& rho,
        Foam::volVectorField& delta_U,
        Foam::volVectorField& delta_U_prev,
        Foam::volScalarField& delta_p_rgh,
        Foam::volScalarField& delta_p_rgh_CFD,
        Foam::volScalarField& delta_p_rgh_prev,
        Foam::volScalarField& p_rgh_prev,
        Foam::volVectorField& delta_delta_U,
        Foam::volVectorField& delta_delta_U_prev,
        Foam::volScalarField& delta_delta_p_rgh,
        Foam::volScalarField& delta_delta_p_rgh_prev,
        Foam::volScalarField& delta_delta_p_rgh_CFD
    );

    ~SurrogateModel();

    void init();
    void predict();
    void reload();
    bool initialized() const { return initialized_; }
};

#endif
