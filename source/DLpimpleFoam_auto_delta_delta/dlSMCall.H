
// Use delta_delta fields for ML surrogate model
// Layout: U(3) + delta_delta_U(3) + delta_delta_U_prev(3) + delta_delta_p_prev(1) + delta_p_prev(1) + delta_U(3) + p_prev(1)
U_MAX_NORM = 0.0;
forAll(delta_delta_p.internalField(), id)
{
    // U field (channels 0-2)
    input_vals[id][0] = U[id].x();
    input_vals[id][1] = U[id].y();
    input_vals[id][2] = U[id].z();

    // delta_delta_U field (channels 3-5)
    input_vals[id][3] = delta_delta_U[id].x();
    input_vals[id][4] = delta_delta_U[id].y();
    input_vals[id][5] = delta_delta_U[id].z();

    // delta_delta_U_prev field (channels 6-8)
    input_vals[id][6] = delta_delta_U_prev[id].x();
    input_vals[id][7] = delta_delta_U_prev[id].y();
    input_vals[id][8] = delta_delta_U_prev[id].z();

    // delta_delta_p_prev (channel 9)
    input_vals[id][9] = delta_delta_p_prev[id];

    // delta_p_prev (channel 10)
    input_vals[id][10] = delta_p_prev[id];

    // delta_U (channels 11-13)
    input_vals[id][11] = delta_U[id].x();
    input_vals[id][12] = delta_U[id].y();
    input_vals[id][13] = delta_U[id].z();

    // p_prev (channel 14)
    input_vals[id][14] = p_prev[id];

    // div_delta_delta_U (channel 15)
    input_vals[id][15] = div_delta_delta_U[id];
    // div_dU (channel 16)
    input_vals[id][16] = div_dU[id];
    // div_U (channel 17)
    input_vals[id][17] = div_U[id];

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

Info << "delta_delta_PValue ndim: " << PyArray_NDIM(delta_delta_PValue) << ", shape: ";
for (int i = 0; i < PyArray_NDIM(delta_delta_PValue); ++i)
    Info << PyArray_DIM(delta_delta_PValue, i) << " ";
Info << " dtype: " << PyArray_TYPE(delta_delta_PValue) << " (should be 12 for NPY_DOUBLE)" << endl;
Info << "First 5 values: ";
for (int i = 0; i < std::min((npy_intp)5, size); ++i)
    Info << data[i] << " ";
Info << endl;

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

double min_val = data[0];
double max_val = data[0];
for (npy_intp i = 1; i < size; i++)
{
    if (data[i] < min_val) min_val = data[i];
    if (data[i] > max_val) max_val = data[i];
}
Info << "delta_delta_PValue min: " << min_val << ", max: " << max_val << endl;

Info<< ">>> Filling delta_delta_p <<<" << endl;

forAll(delta_delta_p.internalField(), id)
{
    delta_delta_p[id] = *((double*)PyArray_GETPTR2(delta_delta_PValue, id, 0));
}

Info<< ">>> delta_delta_p filled <<<" << endl;

Py_DECREF(delta_delta_PValue);

// PyTuple_SetItem stole the reference to array_3d_
array_3d_ = nullptr;
