#include "MLSampling.H"
#include "POSIX.H"
#include <hdf5.h>
#include <cstring>
#include <vector>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

DataSampler::DataSampler
(
    const Foam::fvMesh& mesh,
    Foam::volVectorField& delta_U,
    Foam::volScalarField& delta_p_rgh_CFD,
    const std::string& dataDir,
    const std::string& sourceDir,
    int warmUpSteps,
    int burstSteps,
    int regularInterval,
    int retrainInterval
)
:
    warmUpSteps_(warmUpSteps),
    burstSteps_(burstSteps),
    regularInterval_(regularInterval),
    retrainInterval_(retrainInterval),
    timeStep_(0),
    sampleCount_(0),
    samplesSinceRetrain_(0),
    initialTrainingDone_(false),
    coordinatesWritten_(false),
    masterFile_(-1),
    mesh_(mesh),
    delta_U_(delta_U),
    delta_p_rgh_CFD_(delta_p_rgh_CFD),
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

    // Phase 2: burst sampling — every step for burstSteps_ steps
    int stepsSinceWarmUp = timeStep_ - warmUpSteps_;
    if (stepsSinceWarmUp <= burstSteps_)
    {
        return true;
    }

    // Phase 3: regular interval sampling
    int stepsSinceBurst = timeStep_ - (warmUpSteps_ + burstSteps_);
    return (stepsSinceBurst % regularInterval_ == 0);
}


bool DataSampler::shouldRetrain() const
{
    return (samplesSinceRetrain_ >= retrainInterval_);
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

void DataSampler::writeFieldData
(
    const Foam::volVectorField& vf,
    const Foam::volScalarField& sf,
    int step
)
{
    Foam::Info<< "  [DataSampler] writeFieldData() called for step " << step << Foam::nl;

    std::string masterPath = dataDir_ + "/data.h5";

    if (masterFile_ < 0)
    {
        if (Foam::isFile(masterPath))
        {
            Foam::Info<< "  [DataSampler] Reopening existing HDF5 file: " << masterPath << Foam::nl;
            masterFile_ = H5Fopen(masterPath.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
            if (masterFile_ < 0)
            {
                FatalErrorInFunction
                    << "Failed to reopen HDF5 file: " << masterPath
                    << exit(FatalError);
            }
            Foam::Info<< "  [DataSampler] HDF5 file reopened successfully" << Foam::nl;
            // coordinatesWritten_ stays true — coordinates already in this file
        }
        else
        {
            Foam::Info<< "  [DataSampler] Creating new HDF5 file: " << masterPath << Foam::nl;
            masterFile_ = H5Fcreate(masterPath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            if (masterFile_ < 0)
            {
                FatalErrorInFunction
                    << "Failed to create HDF5 file: " << masterPath
                    << exit(FatalError);
            }
            // File is new — coordinates must be written again
            coordinatesWritten_ = false;
            Foam::Info<< "  [DataSampler] New HDF5 file created, coordinates will be re-written" << Foam::nl;
        }
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

    // Prepare velocity_increment and pressure_increment buffers using vectors
    std::vector<double> velocity_increment(nCells * 3);
    std::vector<double> pressure_increment(nCells);

    forAll(Uvals, i)
    {
        velocity_increment[3*i]   = Uvals[i].x();
        velocity_increment[3*i+1] = Uvals[i].y();
        velocity_increment[3*i+2] = Uvals[i].z();

        pressure_increment[i]     = pvals[i];
    }

    Foam::Info<< "  [DataSampler] Buffers filled" << Foam::nl;

    // Create dataspaces
    hsize_t vel_dims[2]   = {(hsize_t)nCells, 3};
    hsize_t pres_dims[1]  = {(hsize_t)nCells};

    hid_t vel_space   = H5Screate_simple(2, vel_dims, nullptr);
    hid_t pres_space  = H5Screate_simple(1, pres_dims, nullptr);

    // Create datasets within the group
    hid_t vel_dset = H5Dcreate
    (
        group_id, "velocity_increment", H5T_IEEE_F64LE,
        vel_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );
    hid_t pres_dset = H5Dcreate
    (
        group_id, "pressure_increment", H5T_IEEE_F64LE,
        pres_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT
    );

    Foam::Info<< "  [DataSampler] Writing velocity and pressure datasets" << Foam::nl;

    // Write data
    herr_t status = H5Dwrite(vel_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, velocity_increment.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing velocity data" << Foam::nl;
    
    status = H5Dwrite(pres_dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, pressure_increment.data());
    if (status < 0) Foam::Info<< "  [DataSampler] ERROR writing pressure data" << Foam::nl;

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
    H5Dclose(vel_dset);
    H5Dclose(pres_dset);
    H5Sclose(vel_space);
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
    
    writeFieldData(delta_U_, delta_p_rgh_CFD_, timeStep_);

    sampleCount_++;
    samplesSinceRetrain_++;

    Foam::Info<< "  [DataSampler] Sample #" << sampleCount_ << " completed at step " << timeStep_ << Foam::nl;
    Foam::Info<< "  [DataSampler] Samples since retrain: " << samplesSinceRetrain_ << Foam::nl;
}

bool DataSampler::update()
{
    timeStep_++;
    bool activateSurrogate = false;

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
            activateSurrogate = true;
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
                Foam::Info<< "  [DataSampler] Update training completed successfully" << Foam::nl;
            }

            // Reopen HDF5 file for continued sampling
            reopenHDF5File();
        }

        samplesSinceRetrain_ = 0;
    }

    return activateSurrogate;
}