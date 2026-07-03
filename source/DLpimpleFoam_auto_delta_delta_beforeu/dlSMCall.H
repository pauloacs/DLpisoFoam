
// Use delta_delta fields for ML surrogate model
// Layout: U(3) + ddUStar(3) + ddUStarPrev(3) + ddpPrev(1) + dpPrev(1) + dUStar(3) + p_prev(1) + divUFirstPred(1) + dUCorrPrev(3) + ddUCorrPrev(3) + gradDpPrev(3) + laplaceDpPrev(1) + uDotGradDpPrev(1) + gradDpPrevMag(1) + rAU(1) + HbyA(3) + divHbyA(1)
U_MAX_NORM = 0.0;
forAll(ddpML.internalField(), id)
{
    // U field (channels 0-2)
    input_vals[id][0] = U[id].x();
    input_vals[id][1] = U[id].y();
    input_vals[id][2] = U[id].z();

    // ddUStar field (channels 3-5)
    input_vals[id][3] = ddUStar[id].x();
    input_vals[id][4] = ddUStar[id].y();
    input_vals[id][5] = ddUStar[id].z();

    // ddUStarPrev field (channels 6-8)
    input_vals[id][6] = ddUStarPrev[id].x();
    input_vals[id][7] = ddUStarPrev[id].y();
    input_vals[id][8] = ddUStarPrev[id].z();

    // ddpPrev (channel 9)
    input_vals[id][9] = ddpPrev[id];

    // dpPrev (channel 10)
    input_vals[id][10] = dpPrev[id];

    // dUStar (channels 11-13)
    input_vals[id][11] = dUStar[id].x();
    input_vals[id][12] = dUStar[id].y();
    input_vals[id][13] = dUStar[id].z();

    // p_prev (channel 14)
    input_vals[id][14] = p_prev[id];

    // divUFirstPred (channel 15)
    input_vals[id][15] = divUFirstPred[id];

    // dUCorrPrev (channels 16-18)
    input_vals[id][16] = dUCorrPrev[id].x();
    input_vals[id][17] = dUCorrPrev[id].y();
    input_vals[id][18] = dUCorrPrev[id].z();

    // ddUCorrPrev (channels 19-21)
    input_vals[id][19] = ddUCorrPrev[id].x();
    input_vals[id][20] = ddUCorrPrev[id].y();
    input_vals[id][21] = ddUCorrPrev[id].z();

    // gradDpPrev (channels 22-24)
    input_vals[id][22] = grad_dpPrev[id].x();
    input_vals[id][23] = grad_dpPrev[id].y();
    input_vals[id][24] = grad_dpPrev[id].z();

    // Extra scalar channels from dpPrev derivatives
    input_vals[id][25] = laplace_dpPrev[id];
    input_vals[id][26] = uDotGradDpPrev[id];
    input_vals[id][27] = gradDpPrevMag[id];

    // PISO pressure-equation quantities
    // rAU (channel 28)
    input_vals[id][28] = rAU_ML[id];

    // HbyA (channels 29-31)
    input_vals[id][29] = HbyA_ML[id].x();
    input_vals[id][30] = HbyA_ML[id].y();
    input_vals[id][31] = HbyA_ML[id].z();

    // divHbyA (channel 32)
    input_vals[id][32] = divHbyA_ML[id];

    // dHbyA = HbyA - HbyA_prev (channels 33-35)
    input_vals[id][33] = dHbyA_ML[id].x();
    input_vals[id][34] = dHbyA_ML[id].y();
    input_vals[id][35] = dHbyA_ML[id].z();

    // dDivHbyA = divHbyA - divHbyA_prev (channel 36)
    input_vals[id][36] = dDivHbyA_ML[id];

    // rAUGradDpPrev = rAU * grad(dpPrev) (channels 37-39)
    input_vals[id][37] = rAUGradDpPrev_ML[id].x();
    input_vals[id][38] = rAUGradDpPrev_ML[id].y();
    input_vals[id][39] = rAUGradDpPrev_ML[id].z();

    // divRAUGradDpPrev = div(rAU*grad(dpPrev)) (channel 40)
    input_vals[id][40] = divRAUGradDpPrev_ML[id];

    // pressureEqResidualp = div(HbyA) - div(rAU*grad(dpPrev)) (channel 41)
    input_vals[id][41] = pressureEqResidualp_ML[id];

    // rAUGradpPrev = rAU * grad(p_prev) (channels 42-44)
    input_vals[id][42] = rAUGradpPrev_ML[id].x();
    input_vals[id][43] = rAUGradpPrev_ML[id].y();
    input_vals[id][44] = rAUGradpPrev_ML[id].z();

    // divRAUGradpPrev = div(rAU*grad(p_prev)) (channel 45)
    input_vals[id][45] = divRAUGradpPrev_ML[id];

    double u_norm = std::sqrt(
        std::pow(U[id].x(), 2) +
        std::pow(U[id].y(), 2) +
        std::pow(U[id].z(), 2)
    );
    U_MAX_NORM = std::max(U_MAX_NORM, u_norm);
}

// Re-create the numpy array wrapping the updated input_vals buffer
Py_XDECREF(array_3d_);
array_3d_ = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, &input_vals[0]);

Info<< ">>> Setting arguments (delta_delta fields) <<<" << endl;

PyTuple_SetItem(py_args, 0, array_3d_);
PyTuple_SetItem(py_args, 1, PyFloat_FromDouble(U_MAX_NORM));

Info<< ">>> Calling python function <<<" << endl;

PyArrayObject *delta_delta_PValue = reinterpret_cast<PyArrayObject*>(
    PyObject_CallObject(py_func, py_args)
);

bool hasNaN = false;
npy_intp size = PyArray_SIZE(delta_delta_PValue);
double* data = static_cast<double*>(PyArray_DATA(delta_delta_PValue));

for (npy_intp i = 0; i < size; i++)
{
    if (std::isnan(data[i]))
    {
        hasNaN = true;
        break;
    }
}
if (hasNaN)
{
    Info<< "WARNING: NaN values detected in delta_delta_PValue" << endl;
}

// Python returns either:
//   1D / (n,1)  — predict_ddU=False — only ddpML
//   (n,4)       — predict_ddU=True  — [ddpML, ddUx, ddUy, ddUz]
if (PyArray_NDIM(delta_delta_PValue) == 2 && PyArray_DIM(delta_delta_PValue, 1) >= 4)
{
    forAll(ddpML.internalField(), id)
    {
        ddpML[id] = *((double*)PyArray_GETPTR2(delta_delta_PValue, id, 0));
        ddUStarML[id].x() = *((double*)PyArray_GETPTR2(delta_delta_PValue, id, 1));
        ddUStarML[id].y() = *((double*)PyArray_GETPTR2(delta_delta_PValue, id, 2));
        ddUStarML[id].z() = *((double*)PyArray_GETPTR2(delta_delta_PValue, id, 3));
    }
    Info<< ">>> ddpML and ddUStarML filled (predict_ddU mode) <<<" << endl;
}
else
{
    forAll(ddpML.internalField(), id)
    {
        ddpML[id] = *((double*)PyArray_GETPTR2(delta_delta_PValue, id, 0));
    }
    Info<< ">>> ddpML filled (pressure-only mode) <<<" << endl;
}


// ----------------------------------------------------------
// DEBUG PRINTS
// ----------------------------------------------------------
Info<< "ddpML stats:"
    << " min = " << gMin(ddpML.internalField())
    << ", max = " << gMax(ddpML.internalField())
    << ", mean = " << gAverage(ddpML.internalField())
    << endl;

label nPrint = 3;

Info<< "ddpML first " << nPrint << " values: ";
for (label i = 0; i < nPrint; ++i)
{
    Info<< ddpML[i] << " ";
}
Info<< endl;



Py_DECREF(delta_delta_PValue);

// PyTuple_SetItem stole the reference to array_3d_
array_3d_ = nullptr;
