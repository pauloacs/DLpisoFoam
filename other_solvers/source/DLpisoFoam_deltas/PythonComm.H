forAll(delta_p.internalField(), id) // for boundary field use u_.boundaryField()
{
        input_vals[id][0] = delta_U[id].x();
        input_vals[id][1] = delta_U[id].y();
        input_vals[id][2] = C[id].x();
        input_vals[id][3] = C[id].y();
        input_vals[id][4] = delta_U_prev[id].x();
        input_vals[id][5] = delta_U_prev[id].y();
        input_vals[id][6] = delta_p_CFD[id];

        // Calculate the maximum norm of U for the current time step
        // To enable non-dimensionalization of the velocity field
        double u_norm = std::sqrt( std::pow(U[id].x(), 2) + std::pow(U[id].y(), 2) );
        U_MAX_NORM = std::max(U_MAX_NORM, u_norm);
        U_MAX_NORM = std::max(U_MAX_NORM, u_norm);
}


PyObject *array_2d = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, &input_vals[0]);

Info<< ">>> Setting arguments <<<" << endl;

PyTuple_SetItem(py_args, 0, array_2d);
PyTuple_SetItem(py_args, 1, PyFloat_FromDouble(U_MAX_NORM));

Info<< ">>> Calling python function <<<" << endl;

// Call python function
PyArrayObject *deltapValue = reinterpret_cast<PyArrayObject*>
(
PyObject_CallObject(py_func, py_args)
);

Info<< ">>> Filling delta_p <<<" << endl;

forAll (p.internalField(),id)
{
         // Here we assume that pValue has a numpy array of dimension 2 with 1 column only
         delta_p[id] = *((double*)PyArray_GETPTR2(deltapValue , id , 0));
}

Info<< ">>>  delta_p filled <<<" << endl;


Py_DECREF(deltapValue);
