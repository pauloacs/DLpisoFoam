U_MAX_NORM = 0.0;

forAll(delta_p_rgh.internalField(), id) // for boundary field use u_.boundaryField()
{
        input_vals[id][0] = delta_U[id].x();
        input_vals[id][1] = delta_U[id].y();
        input_vals[id][2] = delta_U[id].z();
        input_vals[id][3] = delta_U_prev[id].x();
        input_vals[id][4] = delta_U_prev[id].y();
        input_vals[id][5] = delta_U_prev[id].z();
        input_vals[id][6] = delta_p_rgh_CFD[id];

        // Calculate the maximum norm of U for the current time step
        // To enable non-dimensionalization of the velocity field
        double u_norm = std::sqrt(
            std::pow(U[id].x(), 2) +
            std::pow(U[id].y(), 2) +
            std::pow(U[id].z(), 2)
        );
        U_MAX_NORM = std::max(U_MAX_NORM, u_norm);
}


PyObject *array_3d = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, &input_vals[0]);

Info<< ">>> Setting arguments <<<" << endl;

PyTuple_SetItem(py_args, 0, array_3d);
PyTuple_SetItem(py_args, 1, PyFloat_FromDouble(U_MAX_NORM));

Info<< ">>> Calling python function <<<" << endl;

// Call python function
PyArrayObject *deltaPRghValue = reinterpret_cast<PyArrayObject*>
(
PyObject_CallObject(py_func, py_args)
);

// Check for NaNs in deltaPRghValue
bool hasNaN = false;
npy_intp size = PyArray_SIZE(deltaPRghValue);
double* data = static_cast<double*>(PyArray_DATA(deltaPRghValue));
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
    Info<< "WARNING: NaN values detected in deltaPRghValue" << endl;
}

Info<< ">>> Filling delta_p_rgh <<<" << endl;

forAll (delta_p_rgh.internalField(),id)
{
         // Here we assume that pValue has a numpy array of dimension 2 with 1 column only
         delta_p_rgh[id] = *((double*)PyArray_GETPTR2(deltaPRghValue , id , 0));
}

Info<< ">>>  delta_p_rgh filled <<<" << endl;

Py_DECREF(deltaPRghValue);