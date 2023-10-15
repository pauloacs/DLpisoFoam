forAll(p.internalField(), id) // for boundary field use u_.boundaryField()
{
        input_vals[id][0] = U[id].x();
        input_vals[id][1] = U[id].y();
        input_vals[id][2] = C[id].x();
        input_vals[id][3] = C[id].y();
        input_vals[id][4] = p[id]; // passing also the pressure from previous t to correct some regions
}


// create a new array using 'buffer'
//array_2d = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, reinterpret_cast<void*>(input_vals));

PyObject *array_2d = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, &input_vals[0]);

PyTuple_SetItem(py_args, 0, array_2d);
PyTuple_SetItem(py_args, 1, rank_val);

// Call python function

PyArrayObject *pValue = reinterpret_cast<PyArrayObject*>
(
PyObject_CallObject(py_func, py_args)
);

//pValue = (PyArrayObject*)PyObject_CallObject(py_func, py_args); //Casting to PyArrayObject

forAll (p.internalField(),id)
{
         // Here we assume that pValue has a numpy array of dimension 2 with 1 column only
         p[id] = *((double*)PyArray_GETPTR2(pValue , id , 0));
         out[id] = p[id];
}

Py_DECREF(pValue);