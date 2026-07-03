#ifndef MLSampling_H
#define MLSampling_H

#include "fvCFD.H"
#include <string>
#include <hdf5.h>


class DataSampler
{
    // --- Configuration ---
    bool train_;  // If false, skip all sampling
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
    Foam::volVectorField& dUStar_;
    Foam::volScalarField& dp_;
    // Fields for delta_delta
    Foam::volVectorField& ddUStar_;
    Foam::volVectorField& ddUStarPrev_;
    Foam::volScalarField& ddp_;
    Foam::volScalarField& dpPrev_;
    Foam::volScalarField& ddpPrev_;
    Foam::volVectorField& grad_dpPrev_;
    Foam::volScalarField& laplace_dpPrev_;
    Foam::volScalarField& uDotGradDpPrev_;
    Foam::volScalarField& gradDpPrevMag_;
    Foam::volScalarField& p_prev_;
    Foam::volScalarField& divUFirstPred_;
    Foam::volVectorField& ddUCorr_;  // pressure-correction second velocity increment (training target)
    Foam::volVectorField& dUCorrPrev_;   // previous dUCorr (input feature)
    Foam::volVectorField& ddUCorrPrev_;  // previous ddUCorr (input feature)
    Foam::volScalarField& rAU_ML_;       // 1/A(U) (input feature)
    Foam::volVectorField& HbyA_ML_;      // rAU*H(U) (input feature)
    Foam::volScalarField& divHbyA_ML_;   // div(HbyA) (input feature)
    Foam::volVectorField& dHbyA_ML_;     // HbyA - HbyA_prev (input feature)
    Foam::volScalarField& dDivHbyA_ML_;  // divHbyA - divHbyA_prev (input feature)
    Foam::volVectorField& rAUGradDpPrev_ML_;    // rAU * grad(dpPrev)
    Foam::volScalarField& divRAUGradDpPrev_ML_; // div(rAU*grad(dpPrev))
    Foam::volScalarField& pressureEqResidualp_ML_; // div(HbyA)-div(rAU*grad(dpPrev))
    Foam::volVectorField& rAUGradpPrev_ML_;      // rAU * grad(p_prev)
    Foam::volScalarField& divRAUGradpPrev_ML_;   // div(rAU*grad(p_prev))

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
    DataSampler
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
        const std::string& dataDir = "ML_data",
        const std::string& sourceDir = ".",
        bool train = true,
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
};

#endif
