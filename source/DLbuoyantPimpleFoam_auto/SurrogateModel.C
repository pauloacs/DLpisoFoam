// This translation unit owns the numpy API symbol table
#define SURROGATE_MODEL_IMPL
#include "SurrogateModel.H"
#include <cmath>

namespace
{
    // Must live here (not in header) because import_array1
    // is only available when NO_IMPORT_ARRAY is NOT defined.
    int initNumpy()
    {
        import_array1(-1);
        return 0;
    }
}

SurrogateModel::SurrogateModel
(
    const Foam::fvMesh& mesh,
    Foam::volScalarField& p_rgh,
    Foam::volScalarField& p,
    Foam::volVectorField& U,
    Foam::volScalarField& rho,
    Foam::volVectorField& delta_U,
    Foam::volVectorField& delta_U_prev,
    Foam::volScalarField& delta_p_rgh,
    Foam::volScalarField& delta_p_rgh_CFD
)
:
    mesh_(mesh),
    p_rgh_(p_rgh),
    p_(p),
    U_(U),
    rho_(rho),
    delta_U_(delta_U),
    delta_U_prev_(delta_U_prev),
    delta_p_rgh_(delta_p_rgh),
    delta_p_rgh_CFD_(delta_p_rgh_CFD),
    U_MAX_NORM_(0.0),
    input_vals_init_(nullptr),
    input_vals_(nullptr),
    input_vals_z_top_(nullptr),
    input_vals_z_bot_(nullptr),
    input_vals_y_top_(nullptr),
    input_vals_y_bot_(nullptr),
    input_vals_obst_(nullptr),
    py_func_(nullptr),
    py_args_(nullptr),
    init_func_(nullptr),
    init_args_(nullptr),
    reload_func_(nullptr),
    array_3d_(nullptr),
    array_3d_z_top_(nullptr),
    array_3d_z_bot_(nullptr),
    array_3d_y_top_(nullptr),
    array_3d_y_bot_(nullptr),
    array_3d_obst_(nullptr),
    array_3d_init_(nullptr),
    initialized_(false)
{
    dim_np_[0] = 0;
    dim_np_[1] = 0;
}

SurrogateModel::~SurrogateModel()
{
    Py_XDECREF(py_func_);
    Py_XDECREF(py_args_);
    Py_XDECREF(init_func_);
    Py_XDECREF(init_args_);
    Py_XDECREF(reload_func_);
    // Note: arrays handed to PyTuple_SetItem have ownership stolen,
    // only DECREF array_3d_ which is re-created each predict() call
    Py_XDECREF(array_3d_);

    delete[] input_vals_init_;
    delete[] input_vals_;
    delete[] input_vals_z_top_;
    delete[] input_vals_z_bot_;
    delete[] input_vals_y_top_;
    delete[] input_vals_y_bot_;
    delete[] input_vals_obst_;
}

void SurrogateModel::init()
{
    // --- Local aliases matching names used in dlSMCall_init.H ---
    const Foam::fvMesh& mesh              = mesh_;
    Foam::volScalarField& p               = p_;
    Foam::volVectorField& delta_U         = delta_U_;
    Foam::volVectorField& delta_U_prev    = delta_U_prev_;
    Foam::volScalarField& delta_p_rgh     = delta_p_rgh_;
    Foam::volScalarField& delta_p_rgh_CFD = delta_p_rgh_CFD_;
    const Foam::volVectorField& C         = mesh.C();

    double& U_MAX_NORM                    = U_MAX_NORM_;
    auto& input_vals_init                 = input_vals_init_;
    auto& input_vals                      = input_vals_;
    auto& input_vals_z_top                = input_vals_z_top_;
    auto& input_vals_z_bot                = input_vals_z_bot_;
    auto& input_vals_y_top                = input_vals_y_top_;
    auto& input_vals_y_bot                = input_vals_y_bot_;
    auto& input_vals_obst                 = input_vals_obst_;
    PyObject*& py_func                    = py_func_;
    PyObject*& py_args                    = py_args_;
    PyObject*& init_func                  = init_func_;
    PyObject*& init_args                  = init_args_;
    PyObject*& reload_func                = reload_func_;
    PyObject*& array_3d                   = array_3d_;
    PyObject*& array_3d_z_top             = array_3d_z_top_;
    PyObject*& array_3d_z_bot             = array_3d_z_bot_;
    PyObject*& array_3d_y_top             = array_3d_y_top_;
    PyObject*& array_3d_y_bot             = array_3d_y_bot_;
    PyObject*& array_3d_obst              = array_3d_obst_;
    PyObject*& array_3d_init              = array_3d_init_;

    #include "dlSMCall_init.H"

    // Store the numpy dimension for the per-step input array (row x 7)
    // 'row' and 'col' are defined inside dlSMCall_init.H
    dim_np_[0] = row;
    dim_np_[1] = col;

    initialized_ = true;
}

void SurrogateModel::reload()
{
    if (!initialized_ || !reload_func_)
    {
        Foam::Info<< ">>> reload() called before init() — skipping <<<" << Foam::endl;
        return;
    }

    Foam::Info<< ">>> [SurrogateModel] reload() triggered — calling Python reload_weights() <<<" << Foam::endl;
    PyObject* result = PyObject_CallObject(reload_func_, nullptr);
    if (!result)
    {
        PyErr_Print();
        Foam::Info<< ">>> [SurrogateModel] WARNING: reload_weights() Python call FAILED — surrogate still using old weights <<<" << Foam::endl;
    }
    else
    {
        Py_DECREF(result);
        Foam::Info<< ">>> [SurrogateModel] reload_weights() returned OK — surrogate is now using updated weights <<<" << Foam::endl;
    }
}


void SurrogateModel::predict()
{
    // --- Local aliases matching names used in dlSMCall.H ---
    Foam::volVectorField& U               = U_;
    Foam::volVectorField& delta_U         = delta_U_;
    Foam::volVectorField& delta_U_prev    = delta_U_prev_;
    Foam::volScalarField& delta_p_rgh     = delta_p_rgh_;
    Foam::volScalarField& delta_p_rgh_CFD = delta_p_rgh_CFD_;

    double& U_MAX_NORM                    = U_MAX_NORM_;
    auto& input_vals                      = input_vals_;
    PyObject*& py_func                    = py_func_;
    PyObject*& py_args                    = py_args_;
    npy_intp* dim                         = dim_np_;

    #include "dlSMCall.H"
}

