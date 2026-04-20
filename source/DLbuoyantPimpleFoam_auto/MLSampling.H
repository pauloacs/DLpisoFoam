#ifndef MLSampling_H
#define MLSampling_H

#include "fvCFD.H"
#include <string>
#include <hdf5.h>

class DataSampler
{
    // --- Configuration ---
    int warmUpSteps_;           // Steps before any sampling begins
    int burstSteps_;            // Number of samples to collect during burst phase
    int burstInterval_;         // Steps between samples during burst phase
    int regularInterval_;       // After burst, sample every N steps
    int retrainInterval_;       // Call Python retrain every N samples collected
    int windowFrames_;          // Sliding training window size (number of time frames)
    int waitBeforeResampling_;  // Steps to wait after training before resuming sampling

    // --- State ---
    int timeStep_;
    int sampleCount_;
    int samplesSinceRetrain_;
    int stepsSinceLastTrain_;   // Steps elapsed since last training completed
    bool initialTrainingDone_;
    bool coordinatesWritten_;
    hid_t masterFile_;

    // --- References ---
    const Foam::fvMesh& mesh_;
    Foam::volVectorField& delta_U_;
    Foam::volScalarField& delta_p_rgh_CFD_;

    std::string dataDir_;
    std::string sourceDir_;  // Path to source scripts

    // --- Internal helpers ---
    bool shouldSample() const;
    bool shouldRetrain() const;
    void writeSample();
    void writeFieldData
    (
        const Foam::volVectorField& vf,
        const Foam::volScalarField& sf,
        int step
    );
    void writeCoordinatesAndBoundaries();
    void closeHDF5File();      // Close file before Python access
    void reopenHDF5File();     // Reopen file after Python finishes

public:

    // Constructor with configurable parameters
    DataSampler
    (
        const Foam::fvMesh& mesh,
        Foam::volVectorField& delta_U,
        Foam::volScalarField& delta_p_rgh_CFD,
        const std::string& dataDir = "ML_data",
        const std::string& sourceDir = ".",  // Add source directory parameter
        int warmUpSteps = 10,
        int burstSteps = 10,
        int burstInterval = 1,
        int regularInterval = 5,
        int retrainInterval = 20,
        int windowFrames = 30,
        int waitBeforeResampling = 0
    );

    ~DataSampler();

    // Call once per time step after the PIMPLE loop.
    // Returns: 0 = nothing, 1 = first training done (activate surrogate),
    //          2 = retrain done (reload weights into active surrogate).
    int update();

    // Accessors
    int timeStep() const { return timeStep_; }
    int sampleCount() const { return sampleCount_; }
    bool surrogateReady() const { return initialTrainingDone_; }
};

#endif