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
    Foam::volVectorField& dUStar,
    Foam::volScalarField& dp,
    Foam::volVectorField& ddUStar,
    Foam::volVectorField& ddUStarPrev,
    Foam::volScalarField& ddp,
    Foam::volScalarField& dpPrev,
    Foam::volScalarField& ddpPrev,
    Foam::volVectorField& grad_dpPrev,
    Foam::volScalarField& laplace_dpPrev,
    Foam::volScalarField& uDotGradDpPrev,
    Foam::volScalarField& gradDpPrevMag,
    Foam::volScalarField& p_prev,
    Foam::volScalarField& divUFirstPred,
    Foam::volVectorField& ddUCorr,
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
    Foam::volScalarField& divRAUGradpPrev_ML,
    const std::string& dataDir,
    const std::string& sourceDir,
    bool train,
    int warmUpSteps,
    int burstSteps,
    int burstInterval,
    int regularInterval,
    int retrainInterval,
    int windowFrames,
    int waitBeforeResampling
)
:   train_(train),
    warmUpSteps_(warmUpSteps),
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
    dUStar_(dUStar),
    dp_(dp),
    ddUStar_(ddUStar),
    ddUStarPrev_(ddUStarPrev),
    ddp_(ddp),
    dpPrev_(dpPrev),
    ddpPrev_(ddpPrev),
    grad_dpPrev_(grad_dpPrev),
    laplace_dpPrev_(laplace_dpPrev),
    uDotGradDpPrev_(uDotGradDpPrev),
    gradDpPrevMag_(gradDpPrevMag),
    p_prev_(p_prev),
    divUFirstPred_(divUFirstPred),
    ddUCorr_(ddUCorr),
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
    // If training is disabled, never sample
    if (!train_)
    {
        return false;
    }

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

    // Access previous ddUStar field
    const Foam::vectorField& ddUStarPrev_vals = ddUStarPrev_.internalField();
    const Foam::vectorField& ddUCorr_vals = ddUCorr_.internalField();
    const Foam::vectorField& dUCorrPrev_vals = dUCorrPrev_.internalField();
    const Foam::vectorField& ddUCorrPrev_vals = ddUCorrPrev_.internalField();

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

    // Prepare buffers for U (absolute), ddUStar (raw), ddUStarDiff (difference), and ddp
    std::vector<double> U_buf(nCells * 3);
    std::vector<double> ddUStar_buf(nCells * 3);
    std::vector<double> ddUStarDiff_buf(nCells * 3);
    std::vector<double> ddUCorr_buf(nCells * 3);
    std::vector<double> dUStar_buf(nCells * 3);
    std::vector<double> dUCorrPrev_buf(nCells * 3);
    std::vector<double> ddUCorrPrev_buf(nCells * 3);
    std::vector<double> divUFirstPred_buf(nCells);
    std::vector<double> ddp_buf(nCells);
    std::vector<double> dpPrev_buf(nCells);
    std::vector<double> ddpPrev_buf(nCells);
    std::vector<double> gradDpPrev_buf(nCells * 3);
    std::vector<double> laplaceDpPrev_buf(nCells);
    std::vector<double> uDotGradDpPrev_buf(nCells);
    std::vector<double> gradDpPrevMag_buf(nCells);
    std::vector<double> p_prev_buf(nCells);
    std::vector<double> rAU_buf(nCells);
    std::vector<double> HbyA_buf(nCells * 3);
    std::vector<double> divHbyA_buf(nCells);
    std::vector<double> dHbyA_buf(nCells * 3);
    std::vector<double> dDivHbyA_buf(nCells);
    std::vector<double> rAUGradDpPrev_buf(nCells * 3);
    std::vector<double> divRAUGradDpPrev_buf(nCells);
    std::vector<double> pressureEqResidualp_buf(nCells);
    std::vector<double> rAUGradpPrev_buf(nCells * 3);
    std::vector<double> divRAUGradpPrev_buf(nCells);
    
    // Get previous pressure fields
    const Foam::scalarField& pvals_prev = dpPrev_.internalField();
    const Foam::scalarField& pvals_ddpPrev = ddpPrev_.internalField();
    const Foam::vectorField& gradDpPrev_vals = grad_dpPrev_.internalField();
    const Foam::scalarField& laplaceDpPrev_vals = laplace_dpPrev_.internalField();
    const Foam::scalarField& uDotGradDpPrev_vals = uDotGradDpPrev_.internalField();
    const Foam::scalarField& gradDpPrevMag_vals = gradDpPrevMag_.internalField();
    const Foam::scalarField& p_prev_vals = p_prev_.internalField();
    const Foam::vectorField& dUStar_vals = dUStar_.internalField();
    const Foam::scalarField& divUFirstPred_vals = divUFirstPred_.internalField();
    const Foam::scalarField& rAU_ML_vals = rAU_ML_.internalField();
    const Foam::vectorField& HbyA_ML_vals = HbyA_ML_.internalField();
    const Foam::scalarField& divHbyA_ML_vals = divHbyA_ML_.internalField();
    const Foam::vectorField& dHbyA_ML_vals = dHbyA_ML_.internalField();
    const Foam::scalarField& dDivHbyA_ML_vals = dDivHbyA_ML_.internalField();
    const Foam::vectorField& rAUGradDpPrev_ML_vals = rAUGradDpPrev_ML_.internalField();
    const Foam::scalarField& divRAUGradDpPrev_ML_vals = divRAUGradDpPrev_ML_.internalField();
    const Foam::scalarField& pressureEqResidualp_ML_vals = pressureEqResidualp_ML_.internalField();
    const Foam::vectorField& rAUGradpPrev_ML_vals = rAUGradpPrev_ML_.internalField();
    const Foam::scalarField& divRAUGradpPrev_ML_vals = divRAUGradpPrev_ML_.internalField();

    forAll(Uvals, i)
    {
        // Save absolute velocity
        U_buf[3*i]   = U_vals[i].x();
        U_buf[3*i+1] = U_vals[i].y();
        U_buf[3*i+2] = U_vals[i].z();

        // Save raw ddUStar
        ddUStar_buf[3*i]   = Uvals[i].x();
        ddUStar_buf[3*i+1] = Uvals[i].y();
        ddUStar_buf[3*i+2] = Uvals[i].z();

        // Save difference: ddUStar - ddUStarPrev
        ddUStarDiff_buf[3*i]   = Uvals[i].x() - ddUStarPrev_vals[i].x();
        ddUStarDiff_buf[3*i+1] = Uvals[i].y() - ddUStarPrev_vals[i].y();
        ddUStarDiff_buf[3*i+2] = Uvals[i].z() - ddUStarPrev_vals[i].z();

        // Save pressure-correction second velocity increment (training target)
        ddUCorr_buf[3*i]   = ddUCorr_vals[i].x();
        ddUCorr_buf[3*i+1] = ddUCorr_vals[i].y();
        ddUCorr_buf[3*i+2] = ddUCorr_vals[i].z();

        // Save previous dUCorr and ddUCorr (input features)
        dUCorrPrev_buf[3*i]   = dUCorrPrev_vals[i].x();
        dUCorrPrev_buf[3*i+1] = dUCorrPrev_vals[i].y();
        dUCorrPrev_buf[3*i+2] = dUCorrPrev_vals[i].z();
        ddUCorrPrev_buf[3*i]   = ddUCorrPrev_vals[i].x();
        ddUCorrPrev_buf[3*i+1] = ddUCorrPrev_vals[i].y();
        ddUCorrPrev_buf[3*i+2] = ddUCorrPrev_vals[i].z();

        // Save dUStar (first non-conservative velocity increment)
        dUStar_buf[3*i]   = dUStar_vals[i].x();
        dUStar_buf[3*i+1] = dUStar_vals[i].y();
        dUStar_buf[3*i+2] = dUStar_vals[i].z();

        // Save divergence of predicted velocity
        divUFirstPred_buf[i] = divUFirstPred_vals[i];

        // Extra pressure inputs
        dpPrev_buf[i]      = pvals_prev[i];
        ddpPrev_buf[i]     = pvals_ddpPrev[i];
        gradDpPrev_buf[3*i]     = gradDpPrev_vals[i].x();
        gradDpPrev_buf[3*i + 1] = gradDpPrev_vals[i].y();
        gradDpPrev_buf[3*i + 2] = gradDpPrev_vals[i].z();
        laplaceDpPrev_buf[i] = laplaceDpPrev_vals[i];
        uDotGradDpPrev_buf[i] = uDotGradDpPrev_vals[i];
        gradDpPrevMag_buf[i] = gradDpPrevMag_vals[i];
        p_prev_buf[i] = p_prev_vals[i];

        // PISO pressure-equation quantities
        rAU_buf[i]        = rAU_ML_vals[i];
        HbyA_buf[3*i]     = HbyA_ML_vals[i].x();
        HbyA_buf[3*i + 1] = HbyA_ML_vals[i].y();
        HbyA_buf[3*i + 2] = HbyA_ML_vals[i].z();
        divHbyA_buf[i]    = divHbyA_ML_vals[i];

        // Temporal variations of the PISO quantities
        dHbyA_buf[3*i]     = dHbyA_ML_vals[i].x();
        dHbyA_buf[3*i + 1] = dHbyA_ML_vals[i].y();
        dHbyA_buf[3*i + 2] = dHbyA_ML_vals[i].z();
        dDivHbyA_buf[i]    = dDivHbyA_ML_vals[i];

        // Additional pressure-equation-derived quantities
        rAUGradDpPrev_buf[3*i]     = rAUGradDpPrev_ML_vals[i].x();
        rAUGradDpPrev_buf[3*i + 1] = rAUGradDpPrev_ML_vals[i].y();
        rAUGradDpPrev_buf[3*i + 2] = rAUGradDpPrev_ML_vals[i].z();
        divRAUGradDpPrev_buf[i]    = divRAUGradDpPrev_ML_vals[i];
        pressureEqResidualp_buf[i] = pressureEqResidualp_ML_vals[i];
        rAUGradpPrev_buf[3*i]      = rAUGradpPrev_ML_vals[i].x();
        rAUGradpPrev_buf[3*i + 1]  = rAUGradpPrev_ML_vals[i].y();
        rAUGradpPrev_buf[3*i + 2]  = rAUGradpPrev_ML_vals[i].z();
        divRAUGradpPrev_buf[i]     = divRAUGradpPrev_ML_vals[i];

        // ddp to predict (CFD label)
        ddp_buf[i]     = pvals[i];
    }

    Foam::Info<< "  [DataSampler] Buffers filled (ddUStar and ddUStarDiff)" << Foam::nl;

    // Create dataspaces
    hsize_t vel_dims[2]   = {(hsize_t)nCells, 3};
    hsize_t pres_dims[1]  = {(hsize_t)nCells};

    hid_t dU_space    = H5Screate_simple(2, vel_dims, nullptr);
    hid_t U_space     = H5Screate_simple(2, vel_dims, nullptr);
    hid_t raw_space   = H5Screate_simple(2, vel_dims, nullptr);
    hid_t diff_space  = H5Screate_simple(2, vel_dims, nullptr);
    hid_t ddU_CFD_space = H5Screate_simple(2, vel_dims, nullptr);
    hid_t dUCorrPrev_space  = H5Screate_simple(2, vel_dims, nullptr);
    hid_t ddUCorrPrev_space = H5Screate_simple(2, vel_dims, nullptr);
    hid_t gradDpPrev_space = H5Screate_simple(2, vel_dims, nullptr);
    hid_t pres_space  = H5Screate_simple(1, pres_dims, nullptr);

    // Create datasets within the group
    hid_t U_dset = H5Dcreate
    (
        group_id, "U", H5T_IEEE_F64LE,
        U_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t ddUStar_dset = H5Dcreate
    (
        group_id, "ddUStar", H5T_IEEE_F64LE,
        raw_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t ddUStarDiff_dset = H5Dcreate
    (
        group_id, "ddUStarDiff", H5T_IEEE_F64LE,
        diff_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t ddUCorr_dset = H5Dcreate
    (
        group_id, "ddUCorr", H5T_IEEE_F64LE,
        ddU_CFD_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t pres_dset = H5Dcreate
    (
        group_id, "ddp", H5T_IEEE_F64LE,
        pres_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t pres_prev_dset = H5Dcreate
    (
        group_id, "dpPrev", H5T_IEEE_F64LE,
        pres_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t ddpPrev_dset = H5Dcreate
    (
        group_id, "ddpPrev", H5T_IEEE_F64LE,
        pres_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t gradDpPrev_dset = H5Dcreate
    (
        group_id, "gradDpPrev", H5T_IEEE_F64LE,
        gradDpPrev_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t laplaceDpPrev_dset = H5Dcreate
    (
        group_id, "laplaceDpPrev", H5T_IEEE_F64LE,
        pres_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t uDotGradDpPrev_dset = H5Dcreate
    (
        group_id, "uDotGradDpPrev", H5T_IEEE_F64LE,
        pres_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t gradDpPrevMag_dset = H5Dcreate
    (
        group_id, "gradDpPrevMag", H5T_IEEE_F64LE,
        pres_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t rAU_dset = H5Dcreate
    (
        group_id, "rAU", H5T_IEEE_F64LE,
        pres_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t HbyA_dset = H5Dcreate
    (
        group_id, "HbyA", H5T_IEEE_F64LE,
        gradDpPrev_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t divHbyA_dset = H5Dcreate
    (
        group_id, "divHbyA", H5T_IEEE_F64LE,
        pres_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t dHbyA_dset = H5Dcreate
    (
        group_id, "dHbyA", H5T_IEEE_F64LE,
        gradDpPrev_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t dDivHbyA_dset = H5Dcreate
    (
        group_id, "dDivHbyA", H5T_IEEE_F64LE,
        pres_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t rAUGradDpPrev_dset = H5Dcreate
    (
        group_id, "rAUGradDpPrev", H5T_IEEE_F64LE,
        gradDpPrev_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t divRAUGradDpPrev_dset = H5Dcreate
    (
        group_id, "divRAUGradDpPrev", H5T_IEEE_F64LE,
        pres_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t pressureEqResidualp_dset = H5Dcreate
    (
        group_id, "pressureEqResidualp", H5T_IEEE_F64LE,
        pres_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t rAUGradpPrev_dset = H5Dcreate
    (
        group_id, "rAUGradpPrev", H5T_IEEE_F64LE,
        gradDpPrev_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t divRAUGradpPrev_dset = H5Dcreate
    (
        group_id, "divRAUGradpPrev", H5T_IEEE_F64LE,
        pres_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t dUStar_dset = H5Dcreate
    (
        group_id, "dUStar", H5T_IEEE_F64LE,
        dU_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t p_prev_dset = H5Dcreate
    (
        group_id, "p_prev", H5T_IEEE_F64LE,
        pres_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t divUFirstPred_dset = H5Dcreate
    (
        group_id, "divUFirstPred", H5T_IEEE_F64LE,
        pres_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t dUCorrPrev_dset = H5Dcreate
    (
        group_id, "dUCorrPrev", H5T_IEEE_F64LE,
        dUCorrPrev_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t ddUCorrPrev_dset = H5Dcreate
    (
        group_id, "ddUCorrPrev", H5T_IEEE_F64LE,
        ddUCorrPrev_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );

    Foam::Info<< "  [DataSampler] Writing U, ddUStar, ddUStarDiff, ddUCorr, dUStar, dUCorrPrev, ddUCorrPrev, p_prev, divUFirstPred, and pressure datasets" << Foam::nl;

    // Write data
    herr_t status = H5Dwrite(U_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, U_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing U data" << Foam::nl;

    status = H5Dwrite(ddUStar_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, ddUStar_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing ddUStar data" << Foam::nl;

    status = H5Dwrite(ddUStarDiff_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, ddUStarDiff_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing ddUStarDiff data" << Foam::nl;

    status = H5Dwrite(ddUCorr_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, ddUCorr_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing ddUCorr data" << Foam::nl;

    status = H5Dwrite(pres_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, ddp_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing pressure data" << Foam::nl;

    status = H5Dwrite(pres_prev_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dpPrev_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing dpPrev data" << Foam::nl;

    status = H5Dwrite(ddpPrev_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, ddpPrev_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing ddpPrev data" << Foam::nl;

    status = H5Dwrite(gradDpPrev_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, gradDpPrev_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing gradDpPrev data" << Foam::nl;

    status = H5Dwrite(laplaceDpPrev_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, laplaceDpPrev_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing laplaceDpPrev data" << Foam::nl;

    status = H5Dwrite(uDotGradDpPrev_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, uDotGradDpPrev_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing uDotGradDpPrev data" << Foam::nl;

    status = H5Dwrite(gradDpPrevMag_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, gradDpPrevMag_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing gradDpPrevMag data" << Foam::nl;

    status = H5Dwrite(rAU_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rAU_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing rAU data" << Foam::nl;

    status = H5Dwrite(HbyA_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, HbyA_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing HbyA data" << Foam::nl;

    status = H5Dwrite(divHbyA_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, divHbyA_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing divHbyA data" << Foam::nl;

    status = H5Dwrite(dHbyA_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dHbyA_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing dHbyA data" << Foam::nl;

    status = H5Dwrite(dDivHbyA_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dDivHbyA_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing dDivHbyA data" << Foam::nl;

    status = H5Dwrite(rAUGradDpPrev_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rAUGradDpPrev_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing rAUGradDpPrev data" << Foam::nl;

    status = H5Dwrite(divRAUGradDpPrev_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, divRAUGradDpPrev_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing divRAUGradDpPrev data" << Foam::nl;

    status = H5Dwrite(pressureEqResidualp_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, pressureEqResidualp_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing pressureEqResidualp data" << Foam::nl;

    status = H5Dwrite(rAUGradpPrev_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rAUGradpPrev_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing rAUGradpPrev data" << Foam::nl;

    status = H5Dwrite(divRAUGradpPrev_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, divRAUGradpPrev_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing divRAUGradpPrev data" << Foam::nl;

    status = H5Dwrite(dUStar_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dUStar_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing dUStar data" << Foam::nl;

    status = H5Dwrite(p_prev_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, p_prev_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing p_prev data" << Foam::nl;

    status = H5Dwrite(divUFirstPred_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, divUFirstPred_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing divUFirstPred data" << Foam::nl;

    status = H5Dwrite(dUCorrPrev_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dUCorrPrev_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing dUCorrPrev data" << Foam::nl;

    status = H5Dwrite(ddUCorrPrev_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, ddUCorrPrev_buf.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing ddUCorrPrev data" << Foam::nl;

    // Close new resources
    H5Dclose(U_dset);
    H5Dclose(ddUStar_dset);
    H5Dclose(ddUStarDiff_dset);
    H5Dclose(ddUCorr_dset);
    H5Dclose(pres_dset);
    H5Dclose(pres_prev_dset);
    H5Dclose(ddpPrev_dset);
    H5Dclose(gradDpPrev_dset);
    H5Dclose(laplaceDpPrev_dset);
    H5Dclose(uDotGradDpPrev_dset);
    H5Dclose(gradDpPrevMag_dset);
    H5Dclose(dUStar_dset);
    H5Dclose(p_prev_dset);
    H5Dclose(divUFirstPred_dset);
    H5Dclose(dUCorrPrev_dset);
    H5Dclose(ddUCorrPrev_dset);
    H5Dclose(rAU_dset);
    H5Dclose(HbyA_dset);
    H5Dclose(divHbyA_dset);
    H5Dclose(dHbyA_dset);
    H5Dclose(dDivHbyA_dset);
    H5Dclose(rAUGradDpPrev_dset);
    H5Dclose(divRAUGradDpPrev_dset);
    H5Dclose(pressureEqResidualp_dset);
    H5Dclose(rAUGradpPrev_dset);
    H5Dclose(divRAUGradpPrev_dset);
    H5Sclose(U_space);
    H5Sclose(raw_space);
    H5Sclose(diff_space);
    H5Sclose(ddU_CFD_space);
    H5Sclose(dU_space);
    H5Sclose(dUCorrPrev_space);
    H5Sclose(ddUCorrPrev_space);
    H5Sclose(gradDpPrev_space);
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
    writeFieldData(ddUStar_, ddp_, timeStep_, U_MAX_NORM);

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
