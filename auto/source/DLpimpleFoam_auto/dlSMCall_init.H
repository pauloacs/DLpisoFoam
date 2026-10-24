/*The following stuff is for Python interoperability*/

Py_Initialize();
PyRun_SimpleString("import sys");
PyRun_SimpleString("sys.path.append(\".\")");
PyRun_SimpleString("import mpi4py; mpi4py.rc.initialize = False; mpi4py.rc.finalize = False");
Info<< ">>> Python Initialized <<<" << endl;

// initialize numpy array library (via non-void wrapper)
if (initNumpy() < 0)
{
    FatalErrorInFunction
        << "Failed to import numpy" << Foam::abort(FatalError);
}
Info<< ">>> Numpy lib initialized <<<" << endl;

Info<< ">>> Decoding and importing Python module <<<" << endl;
PyObject *pName = PyUnicode_DecodeFSDefault("python_module");
PyObject *pModule = PyImport_Import(pName);
Py_DECREF(pName);
if (!pModule)
{
    PyErr_Print();
    FatalErrorInFunction
        << "Failed to import python_module."
        << exit(FatalError);
}

Info<< ">>> Reading init_func <<<" << endl;
init_func = PyObject_GetAttrString(pModule, "init_func");
init_args = PyTuple_New(8);

Info<< ">>> Reading py_func <<<" << endl;
py_func = PyObject_GetAttrString(pModule, "py_func");
py_args = PyTuple_New(3);

Info<< ">>> Reading reload_func <<<" << endl;
reload_func = PyObject_GetAttrString(pModule, "reload_weights");
if (!reload_func) { PyErr_Clear(); }
if (!init_func || !py_func || !PyCallable_Check(init_func) || !PyCallable_Check(py_func))
{
    PyErr_Print();
    FatalErrorInFunction
        << "python_module must provide callable init_func and py_func."
        << exit(FatalError);
}

Info<< ">>> Python functions defined <<<" << endl;

if (PyModule_AddIntConstant(pModule, "solver_total_dU_contract", 1) < 0)
{
    PyErr_Print();
    FatalErrorInFunction << "Could not advertise joint dU solver support."
        << exit(FatalError);
}
Py_DECREF(pModule);

array_3d = nullptr;
array_3d_z_top = nullptr;
array_3d_z_bot = nullptr;
array_3d_y_top = nullptr;
array_3d_y_bot = nullptr;
array_3d_obst = nullptr;
array_3d_init = nullptr;

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

const int row = num_cells;
const int col = 49;  // previous 37 + rAUGradDpPrev(3) + divRAUGradDpPrev(1) + pressureEqResidualp(1) + rAUGradpPrev(3) + divRAUGradpPrev(1)
const int col_init = 12;

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
    input_vals_init[id][0] = ddUStar[id].x();
    input_vals_init[id][1] = ddUStar[id].y();
    input_vals_init[id][2] = ddUStar[id].z();
    input_vals_init[id][3] = C[id].x();
    input_vals_init[id][4] = C[id].y();
    input_vals_init[id][5] = C[id].z();
    input_vals_init[id][6] = ddUStarPrev[id].x();
    input_vals_init[id][7] = ddUStarPrev[id].y();
    input_vals_init[id][8] = ddUStarPrev[id].z();
    input_vals_init[id][9]  = ddp[id];
    input_vals_init[id][10] = dpPrev[id];
    input_vals_init[id][11] = mesh.V()[id];
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

npy_intp dim_init_np[] = {row, col_init};
npy_intp dim_np[] = {row, col};
npy_intp dim_z_top_np[] = {row_z_top, col_boundaries};
npy_intp dim_z_bot_np[] = {row_z_bot, col_boundaries};
npy_intp dim_y_top_np[] = {row_y_top, col_boundaries};
npy_intp dim_y_bot_np[] = {row_y_bot, col_boundaries};
npy_intp dim_obst_np[] = {row_obst, col_boundaries};

// create numpy arrays from C++ data
array_3d_init = PyArray_SimpleNewFromData(2, dim_init_np, NPY_DOUBLE, reinterpret_cast<void*>(input_vals_init));
array_3d = PyArray_SimpleNewFromData(2, dim_np, NPY_DOUBLE, reinterpret_cast<void*>(input_vals));
array_3d_z_top = PyArray_SimpleNewFromData(2, dim_z_top_np, NPY_DOUBLE, reinterpret_cast<void*>(input_vals_z_top));
array_3d_z_bot = PyArray_SimpleNewFromData(2, dim_z_bot_np, NPY_DOUBLE, reinterpret_cast<void*>(input_vals_z_bot));
array_3d_y_top = PyArray_SimpleNewFromData(2, dim_y_top_np, NPY_DOUBLE, reinterpret_cast<void*>(input_vals_y_top));
array_3d_y_bot = PyArray_SimpleNewFromData(2, dim_y_bot_np, NPY_DOUBLE, reinterpret_cast<void*>(input_vals_y_bot));
array_3d_obst = PyArray_SimpleNewFromData(2, dim_obst_np, NPY_DOUBLE, reinterpret_cast<void*>(input_vals_obst));

PyTuple_SetItem(init_args, 0, array_3d_init);
PyTuple_SetItem(init_args, 1, array_3d_z_top);
PyTuple_SetItem(init_args, 2, array_3d_z_bot);
PyTuple_SetItem(init_args, 3, array_3d_y_top);
PyTuple_SetItem(init_args, 4, array_3d_y_bot);
PyTuple_SetItem(init_args, 5, array_3d_obst);

label referenceFaceCount = 0;
forAll(p.boundaryField(), patchIndex)
{
    if (p.boundaryField()[patchIndex].fixesValue())
    {
        referenceFaceCount += p.boundaryField()[patchIndex].size();
    }
}
npy_intp referenceDims[] = {referenceFaceCount, 7};
PyObject* referenceFaces = PyArray_SimpleNew(2, referenceDims, NPY_DOUBLE);
if (!referenceFaces)
{
    PyErr_Print();
    FatalErrorInFunction
        << "Failed to allocate mesh pressure-reference geometry."
        << exit(FatalError);
}
double* referenceData = static_cast<double*>
(
    PyArray_DATA(reinterpret_cast<PyArrayObject*>(referenceFaces))
);
label referenceIndex = 0;
forAll(p.boundaryField(), patchIndex)
{
    if (!p.boundaryField()[patchIndex].fixesValue())
    {
        continue;
    }
    const vectorField& faceCentres = mesh.Cf().boundaryField()[patchIndex];
    const vectorField& faceVectors = mesh.Sf().boundaryField()[patchIndex];
    forAll(faceCentres, faceIndex)
    {
        const scalar faceArea = mag(faceVectors[faceIndex]);
        const vector faceNormal = faceVectors[faceIndex]/max(faceArea, VSMALL);
        for (direction component = 0; component < 3; ++component)
        {
            referenceData[7*referenceIndex + component] = faceCentres[faceIndex][component];
            referenceData[7*referenceIndex + 3 + component] = faceNormal[component];
        }
        referenceData[7*referenceIndex + 6] = faceArea;
        ++referenceIndex;
    }
}
PyTuple_SetItem(init_args, 6, referenceFaces);

const boundBox gridBounds(mesh.points(), true);
npy_intp boundsDims[] = {2, 3};
PyObject* meshBounds = PyArray_SimpleNew(2, boundsDims, NPY_DOUBLE);
if (!meshBounds)
{
    PyErr_Print();
    FatalErrorInFunction
        << "Failed to allocate mesh grid bounds." << exit(FatalError);
}
double* boundsData = static_cast<double*>
(
    PyArray_DATA(reinterpret_cast<PyArrayObject*>(meshBounds))
);
for (direction component = 0; component < 3; ++component)
{
    boundsData[component] = gridBounds.min()[component];
    boundsData[3 + component] = gridBounds.max()[component];
}
PyTuple_SetItem(init_args, 7, meshBounds);

PyObject* initResult = PyObject_CallObject(init_func, init_args);
if (!initResult)
{
    PyErr_Print();
    FatalErrorInFunction
        << "Python init_func failed; surrogate was not initialized."
        << exit(FatalError);
}
Py_DECREF(initResult);

U_MAX_NORM = 0.0;
