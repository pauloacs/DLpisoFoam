/*The following stuff is for Python interoperability*/

Py_Initialize();
PyRun_SimpleString("import sys");
PyRun_SimpleString("sys.path.append(\".\")");
Info<< ">>> Python Initialized <<<" << endl;

// initialize numpy array library
import_array1(-1);
//init_numpy();
Info<< ">>> Numpy lib initialized <<<" << endl;

Info<< ">>> Decoding and importing Python module <<<" << endl;
PyObject *pName = PyUnicode_DecodeFSDefault("python_module"); // Python filename
PyObject *pModule = PyImport_Import(pName);
Py_DECREF(pName);


init_func = PyObject_GetAttrString(pModule, "init_func");
init_args = PyTuple_New(6);

py_func = PyObject_GetAttrString(pModule, "py_func");
py_args = PyTuple_New(2);

Info<< ">>> Python functions defined <<<" << endl;

Py_DECREF(pModule);

array_3d = nullptr;
array_3d_z_top = nullptr;
array_3d_z_bot = nullptr;
array_3d_y_top = nullptr;
array_3d_y_bot = nullptr;
array_3d_obst = nullptr;
array_3d_init = nullptr;
//PyArrayObject *pValue;

/*// Placeholder to grab data before sending to Python*/
int num_cells = mesh.cells().size();

label z_top_patchID = mesh.boundaryMesh().findPatchID("top");
int num_cells_z_top = p.boundaryField()[z_top_patchID].size();

label z_bot_patchID = mesh.boundaryMesh().findPatchID("bot");
int num_cells_z_bot = p.boundaryField()[z_bot_patchID].size();

label y_bot_patchID = mesh.boundaryMesh().findPatchID("front");
int num_cells_y_bot = p.boundaryField()[y_bot_patchID].size();

label y_top_patchID = mesh.boundaryMesh().findPatchID("back");
int num_cells_y_top = p.boundaryField()[y_top_patchID].size();

label obst_patchID = mesh.boundaryMesh().findPatchID("obstacle");
int num_cells_obstacle = p.boundaryField()[obst_patchID].size();

/*Done importing Python functionality*/


/*double** input_vals;*/
const int row = num_cells;
const int col = 7;
const int col_init = 10; // 3D: 3 for delta_U, 3 for C, 3 for delta_U_prev, 1 for delta_p_CFD

const int row_y_top = num_cells_y_top;
const int row_y_bot = num_cells_y_bot;
const int row_z_top = num_cells_z_top;
const int row_z_bot = num_cells_z_bot;
const int row_obst = num_cells_obstacle;

const int col_boundaries = 3;


input_vals_init = new double[row][col_init];
input_vals = new double[row][col];
input_vals_z_top = new double[row_z_top][col_boundaries];
input_vals_z_bot = new double[row_z_bot][col_boundaries];
input_vals_y_top = new double[row_y_top][col_boundaries];
input_vals_y_bot = new double[row_y_bot][col_boundaries];
input_vals_obst = new double[row_obst][col_boundaries];


forAll(p.internalField(), id)
{
        input_vals_init[id][0] = delta_U[id].x();
        input_vals_init[id][1] = delta_U[id].y();
        input_vals_init[id][2] = delta_U[id].z();
        input_vals_init[id][3] = C[id].x();
        input_vals_init[id][4] = C[id].y();
        input_vals_init[id][5] = C[id].z();
        input_vals_init[id][6] = delta_U_prev[id].x();
        input_vals_init[id][7] = delta_U_prev[id].y();
        input_vals_init[id][8] = delta_U_prev[id].z();
        input_vals_init[id][9] = delta_p_rgh_CFD[id];
}

forAll(p.boundaryField()[obst_patchID], id)
{
        input_vals_obst[id][0] = C.boundaryField()[obst_patchID][id].x();
        input_vals_obst[id][1] = C.boundaryField()[obst_patchID][id].y();
        input_vals_obst[id][2] = C.boundaryField()[obst_patchID][id].z();
}

forAll(p.boundaryField()[z_top_patchID], id)
{
        input_vals_z_top[id][0] = C.boundaryField()[z_top_patchID][id].x();
        input_vals_z_top[id][1] = C.boundaryField()[z_top_patchID][id].y();
        input_vals_z_top[id][2] = C.boundaryField()[z_top_patchID][id].z();
}

forAll(p.boundaryField()[z_bot_patchID], id)
{
        input_vals_z_bot[id][0] = C.boundaryField()[z_bot_patchID][id].x();
        input_vals_z_bot[id][1] = C.boundaryField()[z_bot_patchID][id].y();
        input_vals_z_bot[id][2] = C.boundaryField()[z_bot_patchID][id].z();
}

forAll(p.boundaryField()[y_top_patchID], id)
{
        input_vals_y_top[id][0] = C.boundaryField()[y_top_patchID][id].x();
        input_vals_y_top[id][1] = C.boundaryField()[y_top_patchID][id].y();
        input_vals_y_top[id][2] = C.boundaryField()[y_top_patchID][id].z();
}

forAll(p.boundaryField()[y_bot_patchID], id)
{
        input_vals_y_bot[id][0] = C.boundaryField()[y_bot_patchID][id].x();
        input_vals_y_bot[id][1] = C.boundaryField()[y_bot_patchID][id].y();
        input_vals_y_bot[id][2] = C.boundaryField()[y_bot_patchID][id].z();
}

// Numpy array dimensions for all boundaries
npy_intp dim_init[] = {row, col_init};
npy_intp dim[] = {row, col};
npy_intp dim_z_top[] = {row_z_top, col_boundaries};
npy_intp dim_z_bot[] = {row_z_bot, col_boundaries};
npy_intp dim_y_top[] = {row_y_top, col_boundaries};
npy_intp dim_y_bot[] = {row_y_bot, col_boundaries};
npy_intp dim_obst[] = {row_obst, col_boundaries};

// create numpy arrays from C++ data
array_3d_init = PyArray_SimpleNewFromData(2, dim_init, NPY_DOUBLE, reinterpret_cast<void*>(input_vals_init));
array_3d = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, reinterpret_cast<void*>(input_vals));
array_3d_z_top = PyArray_SimpleNewFromData(2, dim_z_top, NPY_DOUBLE, reinterpret_cast<void*>(input_vals_z_top));
array_3d_z_bot = PyArray_SimpleNewFromData(2, dim_z_bot, NPY_DOUBLE, reinterpret_cast<void*>(input_vals_z_bot));
array_3d_y_top = PyArray_SimpleNewFromData(2, dim_y_top, NPY_DOUBLE, reinterpret_cast<void*>(input_vals_y_top));
array_3d_y_bot = PyArray_SimpleNewFromData(2, dim_y_bot, NPY_DOUBLE, reinterpret_cast<void*>(input_vals_y_bot));
array_3d_obst = PyArray_SimpleNewFromData(2, dim_obst, NPY_DOUBLE, reinterpret_cast<void*>(input_vals_obst));

// Set up arguments for init_func
PyTuple_SetItem(init_args, 0, array_3d_init);
PyTuple_SetItem(init_args, 1, array_3d_z_top);
PyTuple_SetItem(init_args, 2, array_3d_z_bot);
PyTuple_SetItem(init_args, 3, array_3d_y_top);
PyTuple_SetItem(init_args, 4, array_3d_y_bot);
PyTuple_SetItem(init_args, 5, array_3d_obst);

(void) PyObject_CallObject(init_func, init_args);

// No need to DECREF arrays after PyTuple_SetItem, as ownership is transferred to the tuple
Py_DECREF(array_3d_init);
Py_DECREF(array_3d_z_top);
Py_DECREF(array_3d_z_bot);
Py_DECREF(array_3d_y_top);
Py_DECREF(array_3d_y_bot);
Py_DECREF(array_3d_obst);
Py_DECREF(array_3d);

// Py_DECREF(init_args);

U_MAX_NORM = 0.0;