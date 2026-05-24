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
        delta_U, delta_U_prev,
        delta_p, delta_p_CFD, delta_p_prev, p_prev,
        delta_delta_U, delta_delta_U_prev,
        delta_delta_p, delta_delta_p_prev, delta_delta_p_CFD,
        div_U, div_dU, div_delta_delta_U
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

    DataSampler dataSampler(
        mesh, U, delta_U, delta_p_CFD, delta_delta_U, delta_delta_U_prev, delta_delta_p_CFD, delta_p_prev, delta_delta_p_prev, p_prev, div_U, div_dU, div_delta_delta_U, "ML_data",
        sourceScriptDir,
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
                delta_U_prev = delta_U;
                delta_delta_U_prev = delta_delta_U;
                delta_p_prev = delta_p_CFD;
                delta_delta_p_prev = delta_delta_p_CFD;
            }

            #include "UEqn.H"

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
        delta_p_CFD = p - p_prev;
        delta_delta_p_CFD = delta_p_CFD - delta_p_prev;

        // --- Sampling, writing, and ML training ---
        int retrainStatus = dataSampler.update();
        if (retrainStatus == 1 && !surrogateActive)
        {
            // First training completed: initialise surrogate (loads weights too)
            Info<< "Initializing DL Surrogate Model." << nl;
            surrogate.init();
            surrogateActive = true;
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
