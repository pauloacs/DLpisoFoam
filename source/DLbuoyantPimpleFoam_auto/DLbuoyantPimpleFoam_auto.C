/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2020 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    buoyantPimpleFoam

Description
    Transient solver for buoyant, turbulent flow of compressible fluids for
    ventilation and heat-transfer, with optional mesh motion and
    mesh topology changes.

    Uses the flexible PIMPLE (PISO-SIMPLE) solution for time-resolved and
    pseudo-transient simulations.

\*---------------------------------------------------------------------------*/

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

/*The following stuff is for Python interoperability*/
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL POD_ARRAY_API
#include <numpy/arrayobject.h>

//void init_numpy() {
//  import_array1();
//}

/*Done importing Python functionality*/
#include <cmath>
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    // create argument list
    Foam::argList args(argc, argv, true,true,/*initialise=*/false);
    if (!args.checkRootCase())
    {
        Foam::FatalError.exit();
    }

    // Some time related variables
    struct timespec tw1, tw2;
    double posix_wall;

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

    int timeStepCounter = 0;
    int sampleCounter = 0;
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

        // This is the folder used for the SM train data handling
        autoMlDataDir = "ML_data";
        if (!Foam::isDir(autoMlDataDir))
        {
            Foam::mkDir(autoMlDataDir);
        }


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

            if (pimple.firstPimpleIter())
            {
                // only call the Surrogate Model in the first PIMPLE correction
                clock_gettime(CLOCK_MONOTONIC, &tw1); // POSIX
                // Call the Surrogate Model to predict the pressure field
                #include "dlSMCall.H"
                clock_gettime(CLOCK_MONOTONIC, &tw2); // POSIX
                posix_wall = 1000.0*tw2.tv_sec + 1e-6*tw2.tv_nsec - (1000.0*tw1.tv_sec + 1e-6*tw1.tv_nsec);
                printf("DL pressure prediction & data transport: %.2f ms\n", posix_wall);
                
                int counter = 0;
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
            delta_p_rgh_CFD = p_rgh - p_rgh_prev;
        }

        rho = thermo.rho();

       

        // Sampling and writing logic for delta_U and delta_p
        timeStepCounter++;
        bool doSample = false;
        int firstTimeToSample = 10; // Start sampling after 10 steps to allow initial transients to settle

        if (timeStepCounter == firstTimeToSample) {
            bool firstRun = true;
        }
        // After 10 steps, sample for 20 steps
        if (timeStepCounter > firstTimeToSample && timeStepCounter <= firstTimeToSample + 20) {
            doSample = true;
            sampleCounter++;
        }
        // After 30 steps, sample every 5 steps
        else if (timeStepCounter > firstTimeToSample + 20 && ((timeStepCounter - (firstTimeToSample + 21)) % 5 == 0)) {
            doSample = true;
        }
        if (doSample) {
            #include "writeSimulationData.H"
        }

        // After 20 samples, call Python script
        // Run Python script after every 10 new samples
        static int lastPythonCall = firstTimeToSample + 20;
        if (timeStepCounter == firstTimeToSample + 20) {
            int ret = system(("python init_interpolation_and_tucker.py --data_dir " + autoMlDataDir).c_str());
            if (ret != 0) {
                Info << "Python init_interpolation_and_tucker.py failed!" << endl;
            }
            lastPythonCall = timeStepCounter;
        } else if (timeStepCounter > firstTimeToSample + 20 && (timeStepCounter - lastPythonCall) >= 10 && ((timeStepCounter - (firstTimeToSample + 21)) % 5 == 0)) {
            int ret = system(("python update_and_train_nn.py --data_dir " + autoMlDataDir).c_str());
            if (ret != 0) {
                Info << "Python update_and_train_nn.py failed!" << endl;
            }
            lastPythonCall = timeStepCounter;
        }

        if firstRun {
            Info << "Initializing DL Surrogate Model static parameters." << endl;
            #include "dlSMCall_init.H"
            firstRun = false;
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
