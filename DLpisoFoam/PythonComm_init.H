/*The following stuff is for Python interoperability*/

Py_Initialize();
PyRun_SimpleString("import sys");
PyRun_SimpleString("sys.path.append(\".\")");

// initialize numpy array library
import_array1(-1);
//init_numpy();

PyObject *pName = PyUnicode_DecodeFSDefault("python_module"); // Python filename
PyObject *pModule = PyImport_Import(pName);
Py_DECREF(pName);

PyObject *py_func = PyObject_GetAttrString(pModule, "py_func");
PyObject *py_args = PyTuple_New(2);

PyObject *init_func = PyObject_GetAttrString(pModule, "init_func");
PyObject *init_args = PyTuple_New(4);

Py_DECREF(pModule);

PyObject *array_2d(nullptr);
PyObject *array_2d_top(nullptr);
PyObject *array_2d_obst(nullptr);
//PyArrayObject *pValue;

PyObject *rank_val = PyLong_FromLong(Pstream::myProcNo());

/*// Placeholder to grab data before sending to Python*/
int num_cells = mesh.cells().size();

label top_patchID = mesh.boundaryMesh().findPatchID("top");
int num_cells_top = p.boundaryField()[top_patchID].size();

label obst_patchID = mesh.boundaryMesh().findPatchID("obstacle");
int num_cells_obstacle = p.boundaryField()[obst_patchID].size();

/*Done importing Python functionality*/


/*double** input_vals;*/
const int row = num_cells;
const int col = 5; 

const int row_top = num_cells_top;
const int col_top = 2; 

const int row_obst = num_cells_obstacle;
const int col_obst = 2; 


double(*input_vals)[col]{ new double[row][col] };
double(*input_vals_top)[col_top]{ new double[row_top][col_top] };
double(*input_vals_obst)[col_obst]{ new double[row_obst][col_obst] };


forAll(p.internalField(), id) // for boundary field use u_.boundaryField()
{
        input_vals[id][0] = U[id].x();
        input_vals[id][1] = U[id].y();
        input_vals[id][2] = C[id].x();
        input_vals[id][3] = C[id].y();
        input_vals[id][4] = p[id];
}

forAll(p.boundaryField()[obst_patchID], id)
{
        input_vals_obst[id][0] = C.boundaryField()[obst_patchID][id].x();
        input_vals_obst[id][1] = C.boundaryField()[obst_patchID][id].y();
}

forAll(p.boundaryField()[top_patchID], id) // for boundary field use u_.boundaryField()
{
        input_vals_top[id][0] = C.boundaryField()[top_patchID][id].x();
        input_vals_top[id][1] = C.boundaryField()[top_patchID][id].y();
}

// Numpy array dimensions
npy_intp dim[] = {num_cells, 5};
npy_intp dim_top[] = {num_cells_top, 2};
npy_intp dim_obstacle[] = {num_cells_obstacle, 2};

// create a new array using 'buffer'
array_2d = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, reinterpret_cast<void*>(input_vals));
array_2d_top = PyArray_SimpleNewFromData(2, dim_top, NPY_DOUBLE, reinterpret_cast<void*>(input_vals_top));
array_2d_obst = PyArray_SimpleNewFromData(2, dim_obstacle, NPY_DOUBLE, reinterpret_cast<void*>(input_vals_obst));

PyTuple_SetItem(init_args, 0, array_2d);
PyTuple_SetItem(init_args, 1, array_2d_top);
PyTuple_SetItem(init_args, 2, array_2d_obst);
PyTuple_SetItem(init_args, 3, rank_val);

(void) PyObject_CallObject(init_func, init_args);

Py_DECREF(array_2d);
Py_DECREF(array_2d_top);
Py_DECREF(array_2d_obst);

