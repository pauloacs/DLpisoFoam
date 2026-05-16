#ifndef MLSampling_H
#define MLSampling_H

#include "fvCFD.H"
#include <string>
#include <hdf5.h>


class DataSampler
{
    // --- Configuration ---
    int warmUpSteps_;
    int burstSteps_;
    int burstInterval_;
    int regularInterval_;
    int retrainInterval_;
    int windowFrames_;
    int waitBeforeResampling_;

    // --- State ---
    int timeStep_;
    int sampleCount_;
    int samplesSinceRetrain_;
    int stepsSinceLastTrain_;
    bool initialTrainingDone_;
    bool coordinatesWritten_;
    hid_t masterFile_;

    // --- References ---
    const Foam::fvMesh& mesh_;
    Foam::volVectorField& U_;
    Foam::volVectorField& delta_U_;
    Foam::volScalarField& delta_p_rgh_CFD_;
    // New fields for delta_delta
    Foam::volVectorField& delta_delta_U_;
    Foam::volVectorField& delta_delta_U_prev_;
    Foam::volScalarField& delta_delta_p_rgh_CFD_;
    Foam::volScalarField& delta_p_rgh_prev_;
    Foam::volScalarField& delta_delta_p_rgh_prev_;
    Foam::volScalarField& p_rgh_prev_;
    Foam::volScalarField& div_U_;
    Foam::volScalarField& div_dU_;
    Foam::volScalarField& div_delta_delta_U_;

    std::string dataDir_;
    std::string sourceDir_;

    // --- Internal helpers ---
    bool shouldSample() const;
    bool shouldRetrain() const;
    void writeSample();
    void writeFieldData
    (
        const Foam::volVectorField& vf,
        const Foam::volScalarField& sf,
        int step,
        double U_MAX_NORM
    );
    void writeCoordinatesAndBoundaries();
    void closeHDF5File();
    void reopenHDF5File();

public:
    // Constructor with configurable parameters
    DataSampler
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
        const std::string& dataDir = "ML_data",
        const std::string& sourceDir = ".",
        int warmUpSteps = 10,
        int burstSteps = 10,
        int burstInterval = 1,
        int regularInterval = 5,
        int retrainInterval = 20,
        int windowFrames = 30,
        int waitBeforeResampling = 0
    );

    ~DataSampler();

    int update();
    int timeStep() const { return timeStep_; }
    int sampleCount() const { return sampleCount_; }
    bool surrogateReady() const { return initialTrainingDone_; }
};

#endif