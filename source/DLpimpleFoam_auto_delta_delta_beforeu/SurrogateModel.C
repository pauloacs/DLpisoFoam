// This translation unit owns the numpy API symbol table
#define SURROGATE_MODEL_IMPL
#include "SurrogateModel.H"
#include <cmath>

namespace
{
    int initNumpy()
    {
        import_array1(-1);
        return 0;
    }
}

SurrogateModel::SurrogateModel
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
)
: mesh_(mesh),
  p_(p),
  U_(U),
  dUStar_(dUStar),
  dUStarPrev_(dUStarPrev),
  dpML_(dpML),
  dp_(dp),
  dpPrev_(dpPrev),
  p_prev_(p_prev),
  ddUStar_(ddUStar),
  ddUStarPrev_(ddUStarPrev),
  ddpML_(ddpML),
  ddpPrev_(ddpPrev),
  ddp_(ddp),
  grad_dpPrev_(grad_dpPrev),
  laplace_dpPrev_(laplace_dpPrev),
  uDotGradDpPrev_(uDotGradDpPrev),
  gradDpPrevMag_(gradDpPrevMag),
  ddUStarML_(ddUStarML),
  divUFirstPred_(divUFirstPred),
  dUCorrPrev_(dUCorrPrev),
  ddUCorrPrev_(ddUCorrPrev),
  rAU_ML_(rAU_ML),
  HbyA_ML_(HbyA_ML),
  divHbyA_ML_(divHbyA_ML),
  dHbyA_ML_(dHbyA_ML),
  dDivHbyA_ML_(dDivHbyA_ML),
    rAUGradDpPrev_ML_(rAUGradDpPrev_ML),
    divRAUGradDpPrev_ML_(divRAUGradDpPrev_ML),
    pressureEqResidualp_ML_(pressureEqResidualp_ML),
    rAUGradpPrev_ML_(rAUGradpPrev_ML),
    divRAUGradpPrev_ML_(divRAUGradpPrev_ML),
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
    Foam::volVectorField& ddUStar               = ddUStar_;
    Foam::volVectorField& ddUStarPrev           = ddUStarPrev_;
    Foam::volScalarField& ddpML         = ddpML_;
    Foam::volScalarField& ddp     = ddp_;
    Foam::volScalarField& dpPrev          = dpPrev_;
    Foam::volVectorField& dUStar                = dUStar_;
    Foam::volScalarField& p_prev                = p_prev_;
    Foam::volScalarField& divUFirstPred         = divUFirstPred_;
    Foam::volVectorField& dUCorrPrev            = dUCorrPrev_;
    Foam::volVectorField& ddUCorrPrev           = ddUCorrPrev_;
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
        Foam::Info<< ">>> [SurrogateModel] WARNING: reload_weights() Python call FAILED <<<" << Foam::endl;
    }
    else
    {
        Py_DECREF(result);
        Foam::Info<< ">>> [SurrogateModel] reload_weights() returned OK — surrogate updated <<<" << Foam::endl;
    }
}

void SurrogateModel::predict()
{
    // --- Local aliases matching names used in dlSMCall.H ---
    Foam::volVectorField& U               = U_;
    Foam::volVectorField& ddUStar               = ddUStar_;
    Foam::volVectorField& ddUStarPrev           = ddUStarPrev_;
    Foam::volScalarField& ddpML         = ddpML_;
    Foam::volScalarField& ddpPrev    = ddpPrev_;
    Foam::volScalarField& ddp     = ddp_;
    Foam::volVectorField& grad_dpPrev    = grad_dpPrev_;
    Foam::volScalarField& laplace_dpPrev = laplace_dpPrev_;
    Foam::volScalarField& uDotGradDpPrev = uDotGradDpPrev_;
    Foam::volScalarField& gradDpPrevMag = gradDpPrevMag_;
    Foam::volVectorField& ddUStarML             = ddUStarML_;
    Foam::volScalarField& dpPrev          = dpPrev_;
    Foam::volVectorField& dUStar                = dUStar_;
    Foam::volScalarField& p_prev                = p_prev_;
    Foam::volScalarField& divUFirstPred         = divUFirstPred_;
    Foam::volVectorField& dUCorrPrev            = dUCorrPrev_;
    Foam::volVectorField& ddUCorrPrev           = ddUCorrPrev_;
    Foam::volScalarField& rAU_ML                = rAU_ML_;
    Foam::volVectorField& HbyA_ML               = HbyA_ML_;
    Foam::volScalarField& divHbyA_ML            = divHbyA_ML_;
    Foam::volVectorField& dHbyA_ML              = dHbyA_ML_;
    Foam::volScalarField& dDivHbyA_ML           = dDivHbyA_ML_;
    Foam::volVectorField& rAUGradDpPrev_ML      = rAUGradDpPrev_ML_;
    Foam::volScalarField& divRAUGradDpPrev_ML   = divRAUGradDpPrev_ML_;
    Foam::volScalarField& pressureEqResidualp_ML = pressureEqResidualp_ML_;
    Foam::volVectorField& rAUGradpPrev_ML       = rAUGradpPrev_ML_;
    Foam::volScalarField& divRAUGradpPrev_ML    = divRAUGradpPrev_ML_;

    double& U_MAX_NORM                    = U_MAX_NORM_;
    auto& input_vals                      = input_vals_;
    PyObject*& py_func                    = py_func_;
    PyObject*& py_args                    = py_args_;
    npy_intp* dim                         = dim_np_;

    #include "dlSMCall.H"
}
