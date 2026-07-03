// Main file for DLpimpleFoam_auto_delta_delta
// Based on DLpisoFoam_auto_delta_delta but adapted for pimpleFoam (incompressible, PIMPLE algorithm)
// Surrogate model is called only on the first PIMPLE iteration of every time step.

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "kinematicMomentumTransportModel.H"
#include "pimpleControl.H"
#include "fvOptions.H"
#include "localEulerDdtScheme.H"
#include "fvcSmooth.H"
#include "POSIX.H"

#include <time.h>
#include <string>
#include <fstream>
#include "SurrogateModel.H"
#include "MLSampling.H"

// Macro to bake in source directory at compile time via -DSOLVER_SOURCE_DIR
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    // create argument list
    Foam::argList args(argc, argv, true, true, /*initialise=*/false);
    if (!args.checkRootCase())
    {
        Foam::FatalError.exit();
    }

    #include "postProcess.H"

    #include "createTime.H"
    #include "createMesh.H"
    #include "createDyMControls.H"
    #include "createFields.H"
    #include "initContinuityErrs.H"

    turbulence->validate();

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    // Surrogate model (encapsulates all Python/NumPy state)
    SurrogateModel surrogate
    (
        mesh, p, U,
        dUStar, dUStarPrev,
        dpML, dp, dpPrev, p_prev,
        ddUStar, ddUStarPrev,
        ddpML, ddpPrev, ddp,
        grad_dpPrev,
        laplace_dpPrev,
        uDotGradDpPrev,
        gradDpPrevMag,
        ddUStarML,
        divUFirstPred,
        dUCorrPrev, ddUCorrPrev,
        rAU_ML, HbyA_ML, divHbyA_ML, dHbyA_ML, dDivHbyA_ML,
        rAUGradDpPrev_ML, divRAUGradDpPrev_ML, pressureEqResidualp_ML,
        rAUGradpPrev_ML, divRAUGradpPrev_ML
    );
    bool surrogateActive = false;

    // Source script directory is baked in at compile time via -DSOLVER_SOURCE_DIR
    std::string sourceScriptDir(SOLVER_SOURCE_DIR);
    Info<< "Source script directory: " << sourceScriptDir << nl << endl;

    // Read ML sampling parameters from system/MLSamplingDict
    IOdictionary mlDict
    (
        IOobject
        (
            "MLSamplingDict",
            runTime.system(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );
    int warmUpSteps     = mlDict.lookupOrDefault<int>("warmUpSteps",     10);
    int burstSteps      = mlDict.lookupOrDefault<int>("burstSteps",       10);
    int burstInterval   = mlDict.lookupOrDefault<int>("burstInterval",     1);
    int regularInterval = mlDict.lookupOrDefault<int>("regularInterval",   5);
    int retrainInterval       = mlDict.lookupOrDefault<int>("retrainInterval",       20);
    int windowFrames          = mlDict.lookupOrDefault<int>("windowFrames",           30);
    int waitBeforeResampling  = mlDict.lookupOrDefault<int>("waitBeforeResampling",    0);

    // Read the 'train' parameter from python_module.py
    bool train = true;  // default
    {
        std::ifstream pythonModuleFile("python_module.py");
        if (pythonModuleFile.good())
        {
            std::string line;
            while (std::getline(pythonModuleFile, line))
            {
                // Look for line: train = True or train = False
                if (line.find("train") != std::string::npos && 
                    line.find("=") != std::string::npos &&
                    line.find("#") != 0)  // skip comments
                {
                    if (line.find("train") < line.find("="))
                    {
                        // This line has train variable assignment
                        if (line.find("False") != std::string::npos)
                        {
                            train = false;
                            Foam::Info<< "  [DLpimpleFoam] Read train=False from python_module.py — sampling disabled" << Foam::nl;
                        }
                        else if (line.find("True") != std::string::npos)
                        {
                            train = true;
                            Foam::Info<< "  [DLpimpleFoam] Read train=True from python_module.py — sampling enabled" << Foam::nl;
                        }
                        break;
                    }
                }
            }
            pythonModuleFile.close();
        }
    }

    DataSampler dataSampler(
        mesh, U, dUStar, dp, ddUStar, ddUStarPrev, ddp, dpPrev, ddpPrev, grad_dpPrev, laplace_dpPrev, uDotGradDpPrev, gradDpPrevMag, p_prev, divUFirstPred, ddUCorr, dUCorrPrev, ddUCorrPrev, rAU_ML, HbyA_ML, divHbyA_ML, dHbyA_ML, dDivHbyA_ML, rAUGradDpPrev_ML, divRAUGradDpPrev_ML, pressureEqResidualp_ML, rAUGradpPrev_ML, divRAUGradpPrev_ML, "ML_data",
        sourceScriptDir,
        train,
        warmUpSteps, burstSteps, burstInterval, regularInterval, retrainInterval, windowFrames,
        waitBeforeResampling
    );

    while (runTime.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        if (LTS)
        {
            #include "setRDeltaT.H"
        }
        else
        {
            #include "CourantNo.H"
            #include "setDeltaT.H"
        }

        // --- Pressure-velocity PIMPLE corrector loop
        while (pimple.loop())
        {
            int counter = 0;

            // Store previous time step values (only once per time step, on first PIMPLE iteration)
            if (pimple.firstPimpleIter())
            {
                p_prev = p;
                U_time_prev = U;
                dUStarPrev = dUStar;
                dUCorrPrev = dUCorr;
                ddUStarPrev = ddUStar;
                ddUCorrPrev = ddUCorr;
                dpPrev = dp;
                ddpPrev = ddp;
                grad_dpPrev = grad_dp;

                // Store previous time-step PISO quantities for temporal variations
                HbyA_ML_prev = HbyA_ML;
                divHbyA_ML_prev = divHbyA_ML;
            }


            if (pimple.firstPimpleIter())
            {
                // extrapolate pressure from delta values from last time step
                p = p + dpPrev;
                UStarPrev = U;
            }

            #include "UEqn.H"

            if (pimple.firstPimpleIter()){
                // Storing the velocity increment and divergence fields
                // ONLY at the first PIMPLE iteration
                // (These must be consistent inputs for the ML model)

                UStar = U;
                dUStar = UStar - UStarPrev;
                ddUStar = dUStar - dUStarPrev;
                
                // Compute divergence of predicted velocity for ML surrogate model
                divUFirstPred = fvc::div(UStar);

                // Extra dpPrev-derived scalar inputs for ML
                laplace_dpPrev = fvc::laplacian(dpPrev);
                uDotGradDpPrev = U & grad_dpPrev;
                gradDpPrevMag = mag(grad_dpPrev);

                // PISO pressure-equation quantities (ML inputs), evaluated from the
                // first momentum predictor matrix while UEqn is still in scope.
                rAU_ML = 1.0/UEqn.A();
                HbyA_ML = constrainHbyA(rAU_ML*UEqn.H(), U, p);
                divHbyA_ML = fvc::div(HbyA_ML);

                // Additional pressure-equation-derived ML inputs
                rAUGradDpPrev_ML = rAU_ML * grad_dpPrev;
                divRAUGradDpPrev_ML = fvc::laplacian(rAU_ML, dpPrev);

                // OpenFOAM-like known-part residual of pressure equation:
                // pressureEqResidualpKnown = div(HbyA) - laplacian(rAU, pKnown).
                // Here pKnown is the currently known pressure field at this stage.
                pressureEqResidualp_ML = divHbyA_ML - fvc::laplacian(rAU_ML, p);

                // Same operators but using absolute previous pressure p_prev
                rAUGradpPrev_ML = rAU_ML * fvc::grad(p_prev);
                divRAUGradpPrev_ML = fvc::laplacian(rAU_ML, p_prev);

                // Temporal variations of the PISO quantities (current - previous time step)
                dHbyA_ML = HbyA_ML - HbyA_ML_prev;
                dDivHbyA_ML = divHbyA_ML - divHbyA_ML_prev;

            }

            if (pimple.firstPimpleIter() && surrogateActive)
            {
                struct timespec tw1, tw2;
                double posix_wall;
                clock_gettime(CLOCK_MONOTONIC, &tw1);
                surrogate.predict();
                clock_gettime(CLOCK_MONOTONIC, &tw2);
                posix_wall = 1000.0*tw2.tv_sec + 1e-6*tw2.tv_nsec
                           - (1000.0*tw1.tv_sec + 1e-6*tw1.tv_nsec);
                Info<< "DL pressure prediction: "
                    << posix_wall << " ms" << nl;
            
                // increment with the prediction from the surrogate model
                p = p + ddpML;

                #include "UEqn.H"
            }

            
            // --- Pressure corrector loop
            while (pimple.correct())
            {
                #include "pEqn.H"           
                counter++;
            }

            if (pimple.turbCorr())
            {
                laminarTransport.correct();
                turbulence->correct();
            }
        }

        // Calculate pressure deltas AFTER all PIMPLE iterations are complete
        // These are the cumulative CFD-computed values for the entire time step
        dp = p - p_prev;
        ddp = dp - dpPrev;

        grad_dp = fvc::grad(dp);

        // Calculate pressure-correction velocity increments AFTER all PIMPLE iterations are complete
        // dUCorr  = U - UStar         (velocity gained during pressure solve)
        // ddUCorr = dUCorr - dUCorrPrev  (second difference — training target)
        dUCorr  = U - UStar;
        ddUCorr = dUCorr - dUCorrPrev;

        // --- Sampling, writing, and ML training ---
        int retrainStatus = dataSampler.update();
        
        // Initialize surrogate on first timestep if train=False (no training, just inference)
        if (!train && !surrogateActive)
        {
            Info<< "Initializing DL Surrogate Model (inference mode, train=False)." << nl;
            surrogate.init();
            surrogateActive = true;
        }
        else if (retrainStatus == 1 && !surrogateActive)
        {
            // First training completed: initialise surrogate (loads weights too)
            Info<< "Initializing DL Surrogate Model." << nl;
            surrogate.init();
            surrogateActive = true;
            
            // Apply PIMPLE_SM overrides if they exist in fvSolution
            const dictionary& fvSolDict = mesh.solutionDict();
            
            if (fvSolDict.found("PIMPLE_SM"))
            {
                Info<< "Applying PIMPLE_SM solver settings for surrogate-accelerated phase." << nl;
                
                // Read PIMPLE_SM settings
                const dictionary& pimpleSMDict = fvSolDict.subDict("PIMPLE_SM");
                
                // Get write access to PIMPLE dict for modification
                if (fvSolDict.found("PIMPLE"))
                {
                    dictionary& pimpleDict = 
                        const_cast<dictionary&>(fvSolDict.subDict("PIMPLE"));
                    
                    // Apply each setting from PIMPLE_SM to PIMPLE
                    forAllConstIter(dictionary, pimpleSMDict, iter)
                    {
                        pimpleDict.set(iter().clone(pimpleDict).ptr());
                        Info<< "  Override: " << iter().keyword() << nl;
                    }
                }
                else
                {
                    FatalErrorInFunction
                        << "PIMPLE_SM found but PIMPLE dict not found in fvSolution"
                        << exit(FatalError);
                }
            }
        }
        else if (retrainStatus == 2 && surrogateActive)
        {
            // Incremental retrain completed: reload updated weights into memory
            surrogate.reload();
        }

        runTime.write();

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    Info<< "End\n" << endl;

    return 0;
}

// ************************************************************************* //
