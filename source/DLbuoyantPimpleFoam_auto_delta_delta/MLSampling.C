#include "MLSampling.H"
#include "POSIX.H"
#include <hdf5.h>
#include <cstring>
#include <vector>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

DataSampler::DataSampler
(
    const Foam::fvMesh& mesh,
    Foam::volVectorField& U,
    Foam::volVectorField& delta_U,
    Foam::volScalarField& delta_p_rgh_CFD,
    Foam::volVectorField& delta_delta_U,
    Foam::volVectorField& delta_delta_U_prev,
    Foam::volScalarField& delta_delta_p_rgh_CFD,
    Foam::volScalarField& delta_p_rgh_prev,
    Foam::volScalarField& delta_delta_p_rgh_prev,
    Foam::volScalarField& p_rgh_prev,
    Foam::volScalarField& div_U,
    Foam::volScalarField& div_dU,
    Foam::volScalarField& div_delta_delta_U,
    const std::string& dataDir,
    const std::string& sourceDir,
    int warmUpSteps,
    int burstSteps,
    int burstInterval,
    int regularInterval,
    int retrainInterval,
    int windowFrames,
    int waitBeforeResampling
)
:   warmUpSteps_(warmUpSteps),
    burstSteps_(burstSteps),
    burstInterval_(burstInterval),
    regularInterval_(regularInterval),
    retrainInterval_(retrainInterval),
    windowFrames_(windowFrames),
    waitBeforeResampling_(waitBeforeResampling),
    timeStep_(0),
    sampleCount_(0),
    samplesSinceRetrain_(0),
    stepsSinceLastTrain_(waitBeforeResampling),
    initialTrainingDone_(false),
    coordinatesWritten_(false),
    masterFile_(-1),
    mesh_(mesh),
    U_(U),
    delta_U_(delta_U),
    delta_p_rgh_CFD_(delta_p_rgh_CFD),
    delta_delta_U_(delta_delta_U),
    delta_delta_U_prev_(delta_delta_U_prev),
    delta_delta_p_rgh_CFD_(delta_delta_p_rgh_CFD),
    delta_p_rgh_prev_(delta_p_rgh_prev),
    delta_delta_p_rgh_prev_(delta_delta_p_rgh_prev),
    p_rgh_prev_(p_rgh_prev),
    div_U_(div_U),
    div_dU_(div_dU),
    div_delta_delta_U_(div_delta_delta_U),
    dataDir_(dataDir),
    sourceDir_(sourceDir)
{}


DataSampler::~DataSampler()
{
    if (masterFile_ >= 0)
    {
        H5Fclose(masterFile_);
    }
}

bool DataSampler::shouldSample() const
{
    // Phase 1: warm-up — no sampling
    if (timeStep_ <= warmUpSteps_)
    {
        return false;
    }

    // Wait period after a training completes before resuming sampling
    if (initialTrainingDone_ && stepsSinceLastTrain_ < waitBeforeResampling_)
    {
        return false;
    }

    // Phase 2: burst sampling — every burstInterval_ steps for burstSteps_ samples
    int stepsSinceWarmUp = timeStep_ - warmUpSteps_;
    int burstDuration = burstSteps_ * burstInterval_;
    if (stepsSinceWarmUp <= burstDuration)
    {
        return (stepsSinceWarmUp % burstInterval_ == 0);
    }

    // Phase 3: regular interval sampling
    int stepsSinceBurst = timeStep_ - (warmUpSteps_ + burstDuration);
    return (stepsSinceBurst % regularInterval_ == 0);
}


bool DataSampler::shouldRetrain() const
{
    if (!initialTrainingDone_)
    {
        return (sampleCount_ >= burstSteps_);  // fire after burst is complete
    }
    return (samplesSinceRetrain_ >= retrainInterval_);  // subsequent retrains
}


void DataSampler::writeCoordinatesAndBoundaries()
{
    Foam::Info<< "  [DataSampler] Starting writeCoordinatesAndBoundaries()" << Foam::nl;

    // Get cell center coordinates
    const Foam::vectorField& cc = mesh_.C().internalField();
    const int nCells = cc.size();

    Foam::Info<< "  [DataSampler] nCells = " << nCells << Foam::nl;

    // Prepare cell centers buffer
    std::vector<double> coords(nCells * 3);

    forAll(cc, i)
    {
        coords[3*i]     = cc[i].x();
        coords[3*i+1]   = cc[i].y();
        coords[3*i+2]   = cc[i].z();
    }

    Foam::Info<< "  [DataSampler] Filled coordinates buffer" << Foam::nl;

    // Write cell centers
    hsize_t coord_dims[2] = {(hsize_t)nCells, 3};
    hid_t coord_space = H5Screate_simple(2, coord_dims, nullptr);
    if (coord_space < 0)
    {
        Foam::Info<< "  [DataSampler] ERROR: Failed to create coordinate dataspace" << Foam::nl;
        return;
    }

    hid_t coord_dset = H5Dcreate
    (
        masterFile_, "/coordinates", H5T_IEEE_F64LE,
        coord_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    if (coord_dset < 0)
    {
        Foam::Info<< "  [DataSampler] ERROR: Failed to create coordinate dataset" << Foam::nl;
        H5Sclose(coord_space);
        return;
    }

    herr_t status = H5Dwrite(coord_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, coords.data());
    if (status < 0)
    {
        Foam::Info<< "  [DataSampler] ERROR: Failed to write coordinate data" << Foam::nl;
    }
    else
    {
        Foam::Info<< "  [DataSampler] Successfully wrote coordinates dataset" << Foam::nl;
    }

    H5Dclose(coord_dset);
    H5Sclose(coord_space);

    // --- Write boundary face centers and patch names ---
    const Foam::polyBoundaryMesh& boundaryMesh = mesh_.boundaryMesh();
    const Foam::surfaceVectorField::Boundary& Cfb = mesh_.Cf().boundaryField();
    
    // Count total boundary faces
    int totalBoundaryFaces = 0;
    std::vector<int> patchSizes;
    
    for (int patchI = 0; patchI < boundaryMesh.size(); ++patchI)
    {
        int patchSize = boundaryMesh[patchI].size();
        patchSizes.push_back(patchSize);
        Foam::Info<< "  [DataSampler] Patch " << patchI << " (" << boundaryMesh[patchI].name() 
                  << "): " << patchSize << " faces" << Foam::nl;
        totalBoundaryFaces += patchSize;
    }

    Foam::Info<< "  [DataSampler] Total boundary faces: " << totalBoundaryFaces << Foam::nl;

    if (totalBoundaryFaces == 0)
    {
        Foam::Info<< "  [DataSampler] No boundary faces found" << Foam::nl;
        coordinatesWritten_ = true;
        H5Fflush(masterFile_, H5F_SCOPE_GLOBAL);
        return;
    }

    // Prepare boundary data: coordinates + patch index
    std::vector<double> boundary_coords(totalBoundaryFaces * 3, 0.0);
    std::vector<int> boundary_patches(totalBoundaryFaces, -1);
    std::vector<std::string> patchNames;
    
    int faceIdx = 0;
    for (int patchI = 0; patchI < boundaryMesh.size(); ++patchI)
    {
        const Foam::polyPatch& patch = boundaryMesh[patchI];
        patchNames.push_back(patch.name());
        
        // Get face centers for this patch from boundary field
        const Foam::vectorField& patchFaceCentres = Cfb[patchI];
        
        Foam::Info<< "  [DataSampler] Processing patch " << patchI << " with " 
                  << patchFaceCentres.size() << " face centers" << Foam::nl;
        
        if (patchFaceCentres.size() != patchSizes[patchI])
        {
            Foam::Info<< "  [DataSampler] WARNING: Mismatch in face count for patch " << patchI 
                      << " (expected " << patchSizes[patchI] << ", got " << patchFaceCentres.size() << ")" << Foam::nl;
        }
        
        forAll(patchFaceCentres, faceI)
        {
            if (faceIdx >= totalBoundaryFaces)
            {
                Foam::Info<< "  [DataSampler] ERROR: Buffer overflow, faceIdx=" << faceIdx 
                          << " >= totalBoundaryFaces=" << totalBoundaryFaces << Foam::nl;
                break;
            }
            
            boundary_coords[3*faceIdx]     = patchFaceCentres[faceI].x();
            boundary_coords[3*faceIdx+1]   = patchFaceCentres[faceI].y();
            boundary_coords[3*faceIdx+2]   = patchFaceCentres[faceI].z();
            boundary_patches[faceIdx]      = patchI;
            faceIdx++;
        }
    }

    Foam::Info<< "  [DataSampler] Filled boundary coordinates buffer, final faceIdx = " << faceIdx << Foam::nl;

    if (faceIdx != totalBoundaryFaces)
    {
        Foam::Info<< "  [DataSampler] WARNING: faceIdx (" << faceIdx << ") != totalBoundaryFaces (" 
                  << totalBoundaryFaces << ")" << Foam::nl;
    }

    // Write boundary coordinates
    hsize_t boundary_coord_dims[2] = {(hsize_t)faceIdx, 3};
    hid_t boundary_coord_space = H5Screate_simple(2, boundary_coord_dims, nullptr);
    if (boundary_coord_space < 0)
    {
        Foam::Info<< "  [DataSampler] ERROR: Failed to create boundary coordinate dataspace" << Foam::nl;
        return;
    }

    hid_t boundary_coord_dset = H5Dcreate
    (
        masterFile_, "/boundary_coordinates", H5T_IEEE_F64LE,
        boundary_coord_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    if (boundary_coord_dset < 0)
    {
        Foam::Info<< "  [DataSampler] ERROR: Failed to create boundary coordinate dataset" << Foam::nl;
        H5Sclose(boundary_coord_space);
        return;
    }

    status = H5Dwrite(boundary_coord_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, 
             H5P_DEFAULT, boundary_coords.data());
    if (status < 0)
    {
        Foam::Info<< "  [DataSampler] ERROR: Failed to write boundary coordinate data" << Foam::nl;
    }
    else
    {
        Foam::Info<< "  [DataSampler] Successfully wrote boundary coordinates dataset" << Foam::nl;
    }

    H5Dclose(boundary_coord_dset);
    H5Sclose(boundary_coord_space);

    // Write boundary patch indices
    hsize_t boundary_patch_dims[1] = {(hsize_t)faceIdx};
    hid_t boundary_patch_space = H5Screate_simple(1, boundary_patch_dims, nullptr);
    hid_t boundary_patch_dset = H5Dcreate
    (
        masterFile_, "/boundary_patches", H5T_NATIVE_INT,
        boundary_patch_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );

    status = H5Dwrite(boundary_patch_dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, 
             H5P_DEFAULT, boundary_patches.data());
    if (status < 0)
    {
        Foam::Info<< "  [DataSampler] ERROR: Failed to write boundary patch indices" << Foam::nl;
    }
    else
    {
        Foam::Info<< "  [DataSampler] Successfully wrote boundary patches dataset" << Foam::nl;
    }

    H5Dclose(boundary_patch_dset);
    H5Sclose(boundary_patch_space);

    // Write patch names as attributes
    Foam::Info<< "  [DataSampler] Writing " << patchNames.size() << " patch name attributes" << Foam::nl;
    for (size_t i = 0; i < patchNames.size(); ++i)
    {
        std::string attrName = "patch_" + Foam::name(i);
        hid_t attr_space = H5Screate(H5S_SCALAR);
        hid_t str_type = H5Tcopy(H5T_C_S1);
        H5Tset_size(str_type, patchNames[i].size() + 1);
        
        hid_t attr = H5Acreate
        (
            masterFile_, attrName.c_str(), str_type,
            attr_space, H5P_DEFAULT, H5P_DEFAULT
        );
        H5Awrite(attr, str_type, patchNames[i].c_str());
        
        H5Aclose(attr);
        H5Tclose(str_type);
        H5Sclose(attr_space);
    }

    coordinatesWritten_ = true;
    
    // Flush to ensure data is written to disk
    H5Fflush(masterFile_, H5F_SCOPE_GLOBAL);
    
    Foam::Info<< "  [DataSampler] Cell centers and boundary face centers written to HDF5" << Foam::nl;
}

// Modified to accept U_MAX_NORM and write it to HDF5
void DataSampler::writeFieldData
(
    const Foam::volVectorField& vf,
    const Foam::volScalarField& sf,
    int step,
    double U_MAX_NORM // <--- new argument
)
{
    Foam::Info<< "  [DataSampler] writeFieldData() called for step " << step << Foam::nl;

    std::string masterPath = dataDir_ + "/data.h5";

    if (masterFile_ < 0)
    {
        // Python always deletes data.h5 after each training run, so a file found
        // here is stale (e.g. leftover from a crashed previous run).  Always
        // start a fresh file to avoid "name already exists" collisions.
        if (Foam::isFile(masterPath))
        {
            Foam::Info<< "  [DataSampler] Stale HDF5 file found — truncating: " << masterPath << Foam::nl;
        }
        else
        {
            Foam::Info<< "  [DataSampler] Creating new HDF5 file: " << masterPath << Foam::nl;
        }
        masterFile_ = H5Fcreate(masterPath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (masterFile_ < 0)
        {
            FatalErrorInFunction
                << "Failed to create HDF5 file: " << masterPath
                << exit(FatalError);
        }
        Foam::Info<< "  [DataSampler] HDF5 file ready, coordinates will be written" << Foam::nl;
    }

    // Write coordinates and boundaries only once
    if (!coordinatesWritten_)
    {
        Foam::Info<< "  [DataSampler] Writing coordinates and boundaries (first time)" << Foam::nl;
        writeCoordinatesAndBoundaries();
    }

    // Get field data

    const Foam::vectorField& Uvals = vf.internalField();
    const Foam::scalarField& pvals = sf.internalField();
    const int nCells = Uvals.size();

    // Access previous delta_delta_U field
    const Foam::vectorField& Uvals_prev = delta_delta_U_prev_.internalField();

    Foam::Info<< "  [DataSampler] nCells for sample = " << nCells << Foam::nl;

    // Create group for this sample: /sample_N
    std::string groupName = "/sample_" + Foam::name(step);
    Foam::Info<< "  [DataSampler] Creating group: " << groupName << Foam::nl;

    hid_t group_id = H5Gcreate
    (
        masterFile_, groupName.c_str(),
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );

    if (group_id < 0)
    {
        Foam::Info<< "  [DataSampler] ERROR: Failed to create group " << groupName << Foam::nl;
        return;
    }

    Foam::Info<< "  [DataSampler] Group created successfully, copying field data" << Foam::nl;

    // --- Write U_MAX_NORM as a dataset in the sample group ---
    hsize_t scalar_dims[1] = {1};
    hid_t scalar_space = H5Screate_simple(1, scalar_dims, nullptr);
    hid_t uNorm_dset = H5Dcreate(
        group_id, "U_MAX_NORM", H5T_IEEE_F64LE,
        scalar_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    if (uNorm_dset < 0) {
        Foam::Info<< "  [DataSampler] ERROR: Failed to create U_MAX_NORM dataset" << Foam::nl;
    } else {
        herr_t status = H5Dwrite(uNorm_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &U_MAX_NORM);
        if (status < 0) {
            Foam::Info<< "  [DataSampler] ERROR writing U_MAX_NORM data" << Foam::nl;
        } else {
            Foam::Info<< "  [DataSampler] Wrote U_MAX_NORM = " << U_MAX_NORM << Foam::nl;
        }
        H5Dclose(uNorm_dset);
    }
    H5Sclose(scalar_space);


    // Get absolute velocity field
    const Foam::vectorField& U_vals = U_.internalField();

    // Prepare buffers for U (absolute), delta_delta_U (raw), delta_delta_U_diff (difference), and ddp
    std::vector<double> U_buf(nCells * 3);
    std::vector<double> delta_delta_U_buf(nCells * 3);
    std::vector<double> delta_delta_U_diff_buf(nCells * 3);
    std::vector<double> delta_U_buf(nCells * 3);
    std::vector<double> div_U_buf(nCells);
    std::vector<double> div_dU_buf(nCells);
    std::vector<double> div_delta_delta_U_buf(nCells);
    std::vector<double> ddp(nCells);
    std::vector<double> dp_prev(nCells);
    std::vector<double> ddp_prev(nCells);
    std::vector<double> p_rgh_prev_buf(nCells);
    
    // Get previous pressure fields
    const Foam::scalarField& pvals_prev = delta_p_rgh_prev_.internalField();
    const Foam::scalarField& pvals_ddp_prev = delta_delta_p_rgh_prev_.internalField();
    const Foam::scalarField& p_prev_vals = p_rgh_prev_.internalField();
    const Foam::vectorField& dU_vals = delta_U_.internalField();
    const Foam::scalarField& div_U_vals = div_U_.internalField();
    const Foam::scalarField& div_dU_vals = div_dU_.internalField();
    const Foam::scalarField& div_ddu_vals = div_delta_delta_U_.internalField();

    forAll(Uvals, i)
    {
        // Save absolute velocity
        U_buf[3*i]   = U_vals[i].x();
        U_buf[3*i+1] = U_vals[i].y();
        U_buf[3*i+2] = U_vals[i].z();

        // Save raw delta_delta_U
        delta_delta_U_buf[3*i]   = Uvals[i].x();
        delta_delta_U_buf[3*i+1] = Uvals[i].y();
        delta_delta_U_buf[3*i+2] = Uvals[i].z();

        // Save difference: delta_delta_U - delta_delta_U_prev
        delta_delta_U_diff_buf[3*i]   = Uvals[i].x() - Uvals_prev[i].x();
        delta_delta_U_diff_buf[3*i+1] = Uvals[i].y() - Uvals_prev[i].y();
        delta_delta_U_diff_buf[3*i+2] = Uvals[i].z() - Uvals_prev[i].z();

        // Save delta_U (first velocity increment)
        delta_U_buf[3*i]   = dU_vals[i].x();
        delta_U_buf[3*i+1] = dU_vals[i].y();
        delta_U_buf[3*i+2] = dU_vals[i].z();

        // Save divergence fields
        div_U_buf[i] = div_U_vals[i];
        div_dU_buf[i] = div_dU_vals[i];
        div_delta_delta_U_buf[i] = div_ddu_vals[i];

        // Extra pressure inputs
        dp_prev[i]      = pvals_prev[i];
        ddp_prev[i]     = pvals_ddp_prev[i];
        p_rgh_prev_buf[i] = p_prev_vals[i];

        // ddp to predict
        ddp[i]     = pvals[i];
    }

    Foam::Info<< "  [DataSampler] Buffers filled (delta_delta_U and delta_delta_U_diff)" << Foam::nl;

    // Create dataspaces
    hsize_t vel_dims[2]   = {(hsize_t)nCells, 3};
    hsize_t pres_dims[1]  = {(hsize_t)nCells};

    hid_t dU_space    = H5Screate_simple(2, vel_dims, nullptr);
    hid_t U_space     = H5Screate_simple(2, vel_dims, nullptr);
    hid_t raw_space   = H5Screate_simple(2, vel_dims, nullptr);
    hid_t diff_space  = H5Screate_simple(2, vel_dims, nullptr);
    hid_t pres_space  = H5Screate_simple(1, pres_dims, nullptr);

    // Create datasets within the group
    hid_t U_dset = H5Dcreate
    (
        group_id, "U", H5T_IEEE_F64LE,
        U_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t raw_dset = H5Dcreate
    (
        group_id, "delta_delta_U", H5T_IEEE_F64LE,
        raw_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t diff_dset = H5Dcreate
    (
        group_id, "delta_delta_U_diff", H5T_IEEE_F64LE,
        diff_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t pres_dset = H5Dcreate
    (
        group_id, "ddp", H5T_IEEE_F64LE,
        pres_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t pres_prev_dset = H5Dcreate
    (
        group_id, "dp_prev", H5T_IEEE_F64LE,
        pres_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t ddp_prev_dset = H5Dcreate
    (
        group_id, "ddp_prev", H5T_IEEE_F64LE,
        pres_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t dU_dset = H5Dcreate
    (
        group_id, "dU", H5T_IEEE_F64LE,
        dU_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t p_prev_dset = H5Dcreate
    (
        group_id, "p_prev", H5T_IEEE_F64LE,
        pres_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t div_ddu_dset = H5Dcreate
    (
        group_id, "div_delta_delta_U", H5T_IEEE_F64LE,
        pres_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t div_u_dset = H5Dcreate
    (
        group_id, "div_U", H5T_IEEE_F64LE,
        pres_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t div_du_dset = H5Dcreate
    (
        group_id, "div_dU", H5T_IEEE_F64LE,
        pres_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );

    Foam::Info<< "  [DataSampler] Writing U, delta_delta_U, delta_delta_U_diff, dU, p_prev, div_delta_delta_U, div_U, div_dU, and pressure datasets" << Foam::nl;

    // Write data
    herr_t status = H5Dwrite(U_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, U_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing U data" << Foam::nl;

    status = H5Dwrite(raw_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, delta_delta_U_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing delta_delta_U data" << Foam::nl;

    status = H5Dwrite(diff_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, delta_delta_U_diff_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing delta_delta_U_diff data" << Foam::nl;

    status = H5Dwrite(pres_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, ddp.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing pressure data" << Foam::nl;

    status = H5Dwrite(pres_prev_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dp_prev.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing dp_prev data" << Foam::nl;

    status = H5Dwrite(ddp_prev_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, ddp_prev.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing ddp_prev data" << Foam::nl;

    status = H5Dwrite(dU_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, delta_U_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing dU data" << Foam::nl;

    status = H5Dwrite(p_prev_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, p_rgh_prev_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing p_prev data" << Foam::nl;

    status = H5Dwrite(div_ddu_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, div_delta_delta_U_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing div_delta_delta_U data" << Foam::nl;

    status = H5Dwrite(div_u_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, div_U_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing div_U data" << Foam::nl;

    status = H5Dwrite(div_du_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, div_dU_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing div_dU data" << Foam::nl;

    // Close new resources
    H5Dclose(U_dset);
    H5Dclose(raw_dset);
    H5Dclose(diff_dset);
    H5Dclose(pres_dset);
    H5Dclose(pres_prev_dset);
    H5Dclose(ddp_prev_dset);
    H5Dclose(dU_dset);
    H5Dclose(p_prev_dset);
    H5Dclose(div_ddu_dset);
    H5Dclose(div_u_dset);
    H5Dclose(div_du_dset);
    H5Sclose(U_space);
    H5Sclose(raw_space);
    H5Sclose(diff_space);
    H5Sclose(dU_space);
    H5Sclose(pres_space);

    // Write metadata
    hid_t attr_space = H5Screate(H5S_SCALAR);
    hid_t step_attr = H5Acreate
    (
        group_id, "timestep", H5T_NATIVE_INT,
        attr_space, H5P_DEFAULT, H5P_DEFAULT
    );
    int step_val = step;
    H5Awrite(step_attr, H5T_NATIVE_INT, &step_val);

    Foam::Info<< "  [DataSampler] Timestep attribute written" << Foam::nl;


    // Close resources
    H5Aclose(step_attr);
    H5Sclose(attr_space);
    H5Dclose(pres_dset);
    H5Sclose(pres_space);
    H5Gclose(group_id);

    // Flush after each sample write
    H5Fflush(masterFile_, H5F_SCOPE_GLOBAL);
    
    Foam::Info<< "  [DataSampler] Sample write complete and flushed" << Foam::nl;
}

void DataSampler::closeHDF5File()
{
    if (masterFile_ >= 0)
    {
        Foam::Info<< "  [DataSampler] Closing HDF5 file for Python access" << Foam::nl;
        H5Fclose(masterFile_);
        masterFile_ = -1;
    }
}

void DataSampler::reopenHDF5File()
{
    // No-op: file will be reopened lazily on next writeFieldData() call
    Foam::Info<< "  [DataSampler] HDF5 file will be reopened on next sample write" << Foam::nl;
}

void DataSampler::writeSample()
{
    // Ensure directory exists
    if (!Foam::isDir(dataDir_))
    {
        Foam::Info<< "  [DataSampler] Creating directory: " << dataDir_ << Foam::nl;
        Foam::mkDir(dataDir_);
    }

    Foam::Info<< "  [DataSampler] ===== Writing Sample =====" << Foam::nl;
    Foam::Info<< "  [DataSampler] Step: " << timeStep_ << ", Sample Count (before): " << sampleCount_ << Foam::nl;
    
    // Compute U_MAX_NORM for the current time step
    double U_MAX_NORM = 0.0;
    const Foam::volVectorField& U = mesh_.lookupObject<Foam::volVectorField>("U");
    const Foam::vectorField& Uvals = U.internalField();
    forAll(Uvals, id)
    {
        double u_norm = std::sqrt(
            Foam::magSqr(Uvals[id])
        );
        U_MAX_NORM = std::max(U_MAX_NORM, u_norm);
    }
    // Write delta_delta fields instead of delta fields
    writeFieldData(delta_delta_U_, delta_delta_p_rgh_CFD_, timeStep_, U_MAX_NORM);

    sampleCount_++;
    samplesSinceRetrain_++;

    Foam::Info<< "  [DataSampler] Sample #" << sampleCount_ << " completed at step " << timeStep_ << Foam::nl;
    Foam::Info<< "  [DataSampler] Samples since retrain: " << samplesSinceRetrain_ << Foam::nl;
}

int DataSampler::update()
{
    timeStep_++;
    stepsSinceLastTrain_++;
    // 0 = nothing, 1 = first training done (activate surrogate),
    // 2 = incremental retrain done (reload weights into active surrogate)
    int retrainStatus = 0;

    if (shouldSample())
    {
        Foam::Info<< "  [DataSampler] SAMPLE TRIGGERED at step " << timeStep_ << Foam::nl;
        writeSample();
    }

    if (shouldRetrain())
    {
        Foam::Info<< "  [DataSampler] RETRAIN TRIGGERED: " << samplesSinceRetrain_ 
                  << " samples collected (threshold: " << retrainInterval_ << ")" << Foam::nl;
        
        if (!initialTrainingDone_)
        {
            // Close HDF5 file before Python access
            closeHDF5File();
            
            // First training pass
            Foam::Info
                << "  [DataSampler] Running initial ML training ("
                << sampleCount_ << " samples)." << Foam::nl;

            std::string scriptPath = sourceDir_ + "/train_init.py";
            std::string cmd = "python3 " + scriptPath + " --data_dir " + dataDir_;
            
            Foam::Info<< "  [DataSampler] Script path: " << scriptPath << Foam::nl;
            Foam::Info<< "  [DataSampler] Data dir: " << dataDir_ << Foam::nl;
            Foam::Info<< "  [DataSampler] Command: " << cmd << Foam::nl;
            
            int ret = std::system(cmd.c_str());
            if (ret != 0)
            {
                Foam::Info
                    << "  [DataSampler] WARNING: Python script exited with code "
                    << ret << Foam::nl;
            }
            else
            {
                Foam::Info<< "  [DataSampler] Initial training completed successfully" << Foam::nl;
            }

            // Reopen HDF5 file for continued sampling
            reopenHDF5File();

            initialTrainingDone_ = true;
            stepsSinceLastTrain_ = 0;
            retrainStatus = 1;  // Signal: activate surrogate (init loads weights)
        }
        else
        {
            // Close HDF5 file before Python access
            closeHDF5File();
            
            // Incremental update
            Foam::Info
                << "  [DataSampler] Running ML update ("
                << samplesSinceRetrain_ << " new samples)." << Foam::nl;

            std::string scriptPath = sourceDir_ + "/train_update.py";
            std::string cmd = "python3 " + scriptPath
                + " --data_dir " + dataDir_
                + " --window_frames " + std::to_string(windowFrames_);
            
            Foam::Info<< "  [DataSampler] Script path: " << scriptPath << Foam::nl;
            Foam::Info<< "  [DataSampler] Data dir: " << dataDir_ << Foam::nl;
            Foam::Info<< "  [DataSampler] Command: " << cmd << Foam::nl;
            
            int ret = std::system(cmd.c_str());
            if (ret != 0)
            {
                Foam::Info
                    << "  [DataSampler] WARNING: Python script exited with code "
                    << ret << Foam::nl;
            }
            else
            {
                Foam::Info<< "  [DataSampler] Update training completed successfully" << Foam::nl;
            }

            // Reopen HDF5 file for continued sampling
            reopenHDF5File();

            retrainStatus = 2;  // Signal: reload weights into active surrogate
        }

        samplesSinceRetrain_ = 0;
        stepsSinceLastTrain_ = 0;
    }

    return retrainStatus;
}