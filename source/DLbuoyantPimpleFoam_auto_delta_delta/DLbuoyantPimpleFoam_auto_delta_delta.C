// Main file for DLbuoyantPimpleFoam_auto_delta_delta
// Copy of DLbuoyantPimpleFoam_auto.C, to be adapted for delta_delta_p and delta_delta_u

#include "fvCFD.H"
#include "dynamicFvMesh.H"
#include "fluidThermo.H"
#include "fluidThermoMomentumTransportModel.H"
#include "fluidThermophysicalTransportModel.H"
#include "radiationModel.H"
#include "pimpleControl.H"
#include "pressureControl.H"
#include "CorrectPhi.H"
#include "fvOptions.H"
#include "localEulerDdtScheme.H"
#include "fvcSmooth.H"
#include "POSIX.H"

#include <time.h>
#include <string>
#include "SurrogateModel.H"
#include "MLSampling.H"

// Macro to extract directory from __FILE__ at compile time
// __FILE__ is expanded by preprocessor, then we use substring operations
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    // create argument list
    Foam::argList args(argc, argv, true,true,/*initialise=*/false);
    if (!args.checkRootCase())
    {
        Foam::FatalError.exit();
    }

    #include "postProcess.H"

    //#include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createDynamicFvMesh.H"
    #include "createDyMControls.H"
    #include "initContinuityErrs.H"
    #include "createFields.H"
    #include "createFieldRefs.H"
    #include "createRhoUfIfPresent.H"

    turbulence->validate();

    if (!LTS)
    {
        #include "compressibleCourantNo.H"
        #include "setInitialDeltaT.H"
    }

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

	// Surrogate model (encapsulates all Python/NumPy state)
	SurrogateModel surrogate
	(
		mesh, p_rgh, p, U, rho,
		delta_U, delta_U_prev,
		delta_p_rgh, delta_p_rgh_CFD, delta_p_rgh_prev, p_rgh_prev,
		delta_delta_U, delta_delta_U_prev,
		delta_delta_p_rgh, delta_delta_p_rgh_prev, delta_delta_p_rgh_CFD
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
		mesh, U, delta_U, delta_p_rgh_CFD, delta_delta_U, delta_delta_U_prev, delta_delta_p_rgh_CFD, delta_p_rgh_prev, delta_delta_p_rgh_prev, p_rgh_prev, div_U, div_dU, div_delta_delta_U, "ML_data",
		sourceScriptDir,
		warmUpSteps, burstSteps, burstInterval, regularInterval, retrainInterval, windowFrames,
		waitBeforeResampling
	);

	while (pimple.run(runTime))
	{
		#include "readDyMControls.H"

		// Store divrhoU from the previous mesh so that it can be mapped
		// and used in correctPhi to ensure the corrected phi has the
		// same divergence
		autoPtr<volScalarField> divrhoU;
		if (correctPhi)
		{
			divrhoU = new volScalarField
			(
				"divrhoU",
				fvc::div(fvc::absolute(phi, rho, U))
			);
		}

		if (LTS)
		{
			#include "setRDeltaT.H"
		}
		else
		{
			#include "compressibleCourantNo.H"
			#include "setDeltaT.H"
		}

		runTime++;

		Info<< "Time = " << runTime.timeName() << nl << endl;

		// --- Pressure-velocity PIMPLE corrector loop
		while (pimple.loop())
		{
			int counter = 0;
			if (pimple.firstPimpleIter() || moveMeshOuterCorrectors)
			{
				// Store previous time values
				p_rgh_prev = p_rgh;
				delta_U_prev = delta_U;
				delta_delta_U_prev = delta_delta_U;
				delta_p_rgh_prev = delta_p_rgh_CFD;
				delta_delta_p_rgh_prev = delta_delta_p_rgh_CFD;

				// Store momentum to set rhoUf for introduced faces.
				autoPtr<volVectorField> rhoU;
				if (rhoUf.valid())
				{
					rhoU = new volVectorField("rhoU", rho*U);
				}

				// Do any mesh changes
				mesh.update();

				if (mesh.changing())
				{
					gh = (g & mesh.C()) - ghRef;
					ghf = (g & mesh.Cf()) - ghRef;

					MRF.update();

					if (correctPhi)
					{
						// Calculate absolute flux
						// from the mapped surface velocity
						phi = mesh.Sf() & rhoUf();
						#include "correctPhi.H"

						// Make the fluxes relative to the mesh-motion
						fvc::makeRelative(phi, rho, U);
					}

					if (checkMeshCourantNo)
					{
						#include "meshCourantNo.H"
					}
				}
			}

			if (pimple.firstPimpleIter() && !pimple.simpleRho())
			{
				#include "rhoEqn.H"
			}

			#include "UEqn.H"
			#include "EEqn.H"

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
				turbulence->correct();
				thermophysicalTransport->correct();
			}
		}

		// Calculate pressure deltas AFTER all PIMPLE iterations are complete
		// These are the cumulative CFD-computed values for the entire time step
		delta_p_rgh_CFD = p_rgh - p_rgh_prev;
		delta_delta_p_rgh_CFD = delta_p_rgh_CFD - delta_p_rgh_prev;

		rho = thermo.rho();

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
