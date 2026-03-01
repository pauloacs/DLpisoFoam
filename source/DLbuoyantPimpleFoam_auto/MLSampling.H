#ifndef MLSampling_H
#define MLSampling_H

#include "fvCFD.H"
#include <string>

class DataSampler
{
    // --- Configuration ---
    int warmUpSteps_;        // Steps before any sampling begins
    int burstSteps_;         // Number of consecutive steps to sample after warm-up
    int regularInterval_;    // After burst, sample every N steps
    int retrainInterval_;    // Call Python retrain every N samples collected

    // --- State ---
    int timeStep_;
    int sampleCount_;
    int samplesSinceRetrain_;
    bool initialTrainingDone_;

    // --- References ---
    const Foam::fvMesh& mesh_;
    Foam::volVectorField& delta_U_;
    Foam::volScalarField& delta_p_rgh_;

    std::string dataDir_;

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

public:

    // Constructor with configurable parameters
    DataSampler
    (
        const Foam::fvMesh& mesh,
        Foam::volVectorField& delta_U,
        Foam::volScalarField& delta_p_rgh,
        const std::string& dataDir = "ML_data",
        int warmUpSteps = 10,
        int burstSteps = 20,
        int regularInterval = 5,
        int retrainInterval = 10
    );

    // Call once per time step after the PIMPLE loop.
    // Returns true if surrogate model should be activated.
    bool update();

    // Accessors
    int timeStep() const { return timeStep_; }
    int sampleCount() const { return sampleCount_; }
    bool surrogateReady() const { return initialTrainingDone_; }
};

#endif