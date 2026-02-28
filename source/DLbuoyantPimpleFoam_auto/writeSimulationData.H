// writeTrainData.C
// Handles all sampling and writing logic for training data

if (firstRun) {

    // Write cell centers as CSV
    std::ofstream cellCentresStream((autoMlDataDir + "cell_centres.csv").c_str());
    cellCentresStream << "x,y,z\n";
    const pointField& cellCentres = mesh.C();
    forAll(cellCentres, i) {
        cellCentresStream << cellCentres[i].x() << "," << cellCentres[i].y() << "," << cellCentres[i].z() << "\n";
    }
    cellCentresStream.close();

    // Write boundary points as CSV
    std::ofstream boundaryStream((autoMlDataDir + "boundary_points.csv").c_str());
    boundaryStream << "patch,x,y,z\n";
    const polyBoundaryMesh& boundaries = mesh.boundaryMesh();
    forAll(boundaries, patchI) {
        const polyPatch& patch = boundaries[patchI];
        const labelList& faceCells = patch.faceCells();
        forAll(faceCells, i) {
            label cellI = faceCells[i];
            const point& c = mesh.C()[cellI];
            boundaryStream << patch.name() << "," << c.x() << "," << c.y() << "," << c.z() << "\n";
        }
    }
    boundaryStream.close();

    std::ofstream meta((autoMlDataDir + "n_cells.txt").c_str());
    meta << nCells << std::endl;
    meta.close();
    firstRun = false;
}

// Write delta_U and delta_p_rgh to raw binary files for Python (current sample)
std::vector<float> deltaUx(nCells), deltaUy(nCells), deltaUz(nCells), deltaP(nCells);
for (label i = 0; i < nCells; ++i) {
    const vector& u = delta_U.internalField()[i];
    deltaUx[i] = static_cast<float>(u.x());
    deltaUy[i] = static_cast<float>(u.y());
    deltaUz[i] = static_cast<float>(u.z());
    deltaP[i]  = static_cast<float>(delta_p_rgh.internalField()[i]);
}
std::string curDeltaUx = autoMlDataDir + "delta_Ux_" + std::to_string(timeStepCounter) + ".bin";
std::string curDeltaUy = autoMlDataDir + "delta_Uy_" + std::to_string(timeStepCounter) + ".bin";
std::string curDeltaUz = autoMlDataDir + "delta_Uz_" + std::to_string(timeStepCounter) + ".bin";
std::string curDeltaP  = autoMlDataDir + "delta_p_rgh_" + std::to_string(timeStepCounter) + ".bin";
std::ofstream outUx(curDeltaUx, std::ios::binary);
std::ofstream outUy(curDeltaUy, std::ios::binary);
std::ofstream outUz(curDeltaUz, std::ios::binary);
std::ofstream outP(curDeltaP, std::ios::binary);
outUx.write(reinterpret_cast<const char*>(deltaUx.data()), nCells * sizeof(float));
outUy.write(reinterpret_cast<const char*>(deltaUy.data()), nCells * sizeof(float));
outUz.write(reinterpret_cast<const char*>(deltaUz.data()), nCells * sizeof(float));
outP.write(reinterpret_cast<const char*>(deltaP.data()), nCells * sizeof(float));
outUx.close();
outUy.close();
outUz.close();
outP.close();
