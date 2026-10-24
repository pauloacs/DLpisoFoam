// DLpimpleFoam_oracleTwin
//
// Self-sufficient "oracle ceiling" diagnostic solver. No ML, no Python, no
// external training data. Every real timestep is run TWICE:
//
//   Pass 1 (reference): plain PIMPLE, exactly like pimpleFoam. This both
//     advances the simulation AND produces the ground-truth converged
//     pressure for this timestep (dpTruth/ddpTruth).
//
//   Pass 2 (oracle): the full state is rewound to the start of the same
//     timestep, then a perfect pressure guess derived from the just-computed
//     truth (optionally passed through a coarse-grid round trip, and
//     formulated as either 'dp' or 'ddp') is injected immediately before the
//     first pressure equation is assembled — mirroring exactly where the
//     production surrogate solvers inject their prediction. Pass 2's result
//     is never kept; only its GAMG iteration/residual diagnostics are used.
//
// After both passes, state is restored to pass 1's converged result so the
// simulation advances along the real (unperturbed) trajectory.
//
// Config: system/oracleDict
//   integrationMode   frozenPressureEquation | rerunUEqnWithOraclePressure |
//                     directPressureForceCorrection |
//                     pressureForceThenReconstruct;
//   useInterpolation  true | false;
//   gridRes           (dx dy dz);
//   stateReconstructionCorrectors 0 | positive integer;
//   adaptiveStateReconstruction true | false;
//   minStateReconstructionCorrectors positive integer;
//   stateReconstructionResidualTolerance positive scalar;
//   stateReconstructionEquationRelaxation scalar in (0, 1];
//   aitkenStateReconstruction true | false;
//   aitkenMinFactor positive scalar;
//   aitkenMaxFactor scalar >= aitkenMinFactor;

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "kinematicMomentumTransportModel.H"
#include "pimpleControl.H"
#include "fvOptions.H"
#include "fvcSmooth.H"
#include "OracleGrid.H"
#include "Residuals.H"

#include <chrono>

template<class Type>
void clearOracleTwinResiduals(const Foam::polyMesh& mesh)
{
    const Foam::List<Foam::word> fieldNames =
        Foam::Residuals<Type>::fieldNames(mesh);

    forAll(fieldNames, fieldi)
    {
        const_cast
        <
            Foam::DynamicList<Foam::SolverPerformance<Type>>&
        >
        (
            Foam::Residuals<Type>::field(mesh, fieldNames[fieldi])
        ).clear();
    }
}

template<class Type>
Foam::label oracleTwinResidualCount
(
    const Foam::polyMesh& mesh,
    const Foam::word& fieldName
)
{
    return Foam::Residuals<Type>::found(mesh, fieldName)
        ? Foam::Residuals<Type>::field(mesh, fieldName).size()
        : 0;
}

template<class Type>
Foam::scalar oracleTwinFirstResidualSince
(
    const Foam::polyMesh& mesh,
    const Foam::word& fieldName,
    const Foam::label firstSolve
)
{
    const Foam::DynamicList<Foam::SolverPerformance<Type>>& residuals =
        Foam::Residuals<Type>::field(mesh, fieldName);

    return firstSolve < residuals.size()
        ? Foam::cmptMax(residuals[firstSolve].initialResidual())
        : Foam::GREAT;
}

// Apply the existing scalar oracle-grid round trip component-by-component
// to a vector field. This preserves exactly the dp scatter/gather path for dU.
Foam::tmp<Foam::volVectorField> oracleVectorGridRoundTrip
(
    const Foam::volVectorField& fieldIn,
    const Foam::vector& gridRes
)
{
    Foam::tmp<Foam::volVectorField> tFieldOut
    (
        new Foam::volVectorField
        (
            Foam::IOobject
            (
                "oracleVectorGridRoundTrip",
                fieldIn.time().timeName(),
                fieldIn.mesh(),
                Foam::IOobject::NO_READ,
                Foam::IOobject::NO_WRITE,
                false
            ),
            fieldIn
        )
    );

    Foam::volVectorField& fieldOut = tFieldOut.ref();

    for (Foam::direction cmpt = 0; cmpt < Foam::vector::nComponents; ++cmpt)
    {
        Foam::tmp<Foam::volScalarField> tComponentIn =
            fieldIn.component(cmpt);

        Foam::tmp<Foam::volScalarField> tComponentOut =
            oracleGridRoundTrip(tComponentIn(), gridRes);

        fieldOut.replace(cmpt, tComponentOut());
    }

    return tFieldOut;
}

int main(int argc, char *argv[])
{
    #include "postProcess.H"

    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createFields.H"
    #include "initContinuityErrs.H"
    #include "createTimeControls.H"

    turbulence->validate();

    // This diagnostic solver targets fixed-timestep transient PIMPLE cases
    // only; LTS pseudo-transient stepping is not supported (no setRDeltaT.H
    // shipped here since it is never needed).
    #include "CourantNo.H"
    #include "setInitialDeltaT.H"

    IOdictionary oracleDict
    (
        IOobject
        (
            "oracleDict",
            runTime.system(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );
    const word oracleIntegrationMode = oracleDict.lookupOrDefault<word>
    (
        "integrationMode",
        "frozenPressureEquation"
    );
    const bool oracleUseInterpolation =
        oracleDict.lookupOrDefault<bool>("useInterpolation", false);
    const bool oracleUseVelocityIncrement =
        oracleDict.lookupOrDefault<bool>("useVelocityIncrement", false);
    const vector oracleGridRes =
        oracleDict.lookupOrDefault<vector>("gridRes", vector(0.05, 0.05, 0.05));
    const scalar oracleOutputWeightFactor =
        oracleDict.lookupOrDefault<scalar>("outputWeightFactor", 1.0);
    const label oracleStateReconstructionCorrectors =
        oracleDict.lookupOrDefault<label>
        (
            "stateReconstructionCorrectors",
            0
        );
    const bool oracleAdaptiveStateReconstruction =
        oracleDict.lookupOrDefault<bool>
        (
            "adaptiveStateReconstruction",
            false
        );
    const label oracleMinStateReconstructionCorrectors =
        oracleDict.lookupOrDefault<label>
        (
            "minStateReconstructionCorrectors",
            1
        );
    const scalar oracleStateReconstructionResidualTolerance =
        oracleDict.lookupOrDefault<scalar>
        (
            "stateReconstructionResidualTolerance",
            1e-4
        );
    const scalar oracleStateReconstructionEquationRelaxation =
        oracleDict.lookupOrDefault<scalar>
        (
            "stateReconstructionEquationRelaxation",
            1.0
        );
    const bool oracleAitkenStateReconstruction =
        oracleDict.lookupOrDefault<bool>
        (
            "aitkenStateReconstruction",
            false
        );
    const scalar oracleAitkenMinFactor =
        oracleDict.lookupOrDefault<scalar>("aitkenMinFactor", 0.2);
    const scalar oracleAitkenMaxFactor =
        oracleDict.lookupOrDefault<scalar>("aitkenMaxFactor", 1.5);
    const dictionary& pimpleResidualControl = mesh.solutionDict()
        .subDict("PIMPLE").subDict("residualControl");
    const scalar outerUResidualTolerance =
        readScalar(pimpleResidualControl.lookup("U"));
    const scalar outerPResidualTolerance =
        readScalar(pimpleResidualControl.lookup("p"));

    // Allow early termination of the inner PISO pressure-corrector loop.
    // The residual is the initial residual of the first pressure solve in
    // the most recently completed pEqn.H corrector.
    const bool adaptivePressureCorrectors =
        oracleDict.lookupOrDefault<bool>
        (
            "adaptivePressureCorrectors",
            true
        );
    const label minPressureCorrectors =
        oracleDict.lookupOrDefault<label>
        (
            "minPressureCorrectors",
            3
        );
    const scalar pressureCorrectorTolerance =
        oracleDict.lookupOrDefault<scalar>
        (
            "pressureCorrectorTolerance",
            outerPResidualTolerance
        );

    if (minPressureCorrectors < 1)
    {
        FatalErrorInFunction
            << "system/oracleDict: minPressureCorrectors must be at least 1."
            << exit(FatalError);
    }

    if (pressureCorrectorTolerance <= 0)
    {
        FatalErrorInFunction
            << "system/oracleDict: pressureCorrectorTolerance must be positive."
            << exit(FatalError);
    }

    if (oracleStateReconstructionCorrectors < 0)
    {
        FatalErrorInFunction
            << "system/oracleDict: stateReconstructionCorrectors must be "
            << "non-negative." << exit(FatalError);
    }

    if
    (
          oracleAdaptiveStateReconstruction
      &&
          (
                oracleMinStateReconstructionCorrectors < 1
            || oracleMinStateReconstructionCorrectors
                > oracleStateReconstructionCorrectors
          )
    )
    {
        FatalErrorInFunction
            << "system/oracleDict: minStateReconstructionCorrectors must be "
            << "between 1 and stateReconstructionCorrectors."
            << exit(FatalError);
    }

    if (oracleStateReconstructionResidualTolerance <= 0)
    {
        FatalErrorInFunction
            << "system/oracleDict: stateReconstructionResidualTolerance "
            << "must be positive." << exit(FatalError);
    }

    if
    (
        oracleStateReconstructionEquationRelaxation <= 0
     || oracleStateReconstructionEquationRelaxation > 1
    )
    {
        FatalErrorInFunction
            << "system/oracleDict: "
            << "stateReconstructionEquationRelaxation must be in (0, 1]."
            << exit(FatalError);
    }

    if
    (
        oracleAitkenMinFactor <= 0
     || oracleAitkenMaxFactor < oracleAitkenMinFactor
    )
    {
        FatalErrorInFunction
            << "system/oracleDict: require 0 < aitkenMinFactor <= "
            << "aitkenMaxFactor." << exit(FatalError);
    }

    if
    (
        oracleIntegrationMode != "frozenPressureEquation"
     && oracleIntegrationMode != "rerunUEqnWithOraclePressure"
     && oracleIntegrationMode != "directPressureForceCorrection"
     && oracleIntegrationMode != "pressureForceThenReconstruct"
    )
    {
        FatalErrorInFunction
            << "system/oracleDict: unknown integrationMode '"
            << oracleIntegrationMode << "'. Valid modes are "
            << "frozenPressureEquation, rerunUEqnWithOraclePressure, and "
            << "directPressureForceCorrection, and "
            << "pressureForceThenReconstruct." << exit(FatalError);
    }

    Info<< "[ORACLE TWIN] oracle=dp_final"
        << " integrationMode=" << oracleIntegrationMode
        << " useInterpolation=" << oracleUseInterpolation
        << " useVelocityIncrement=" << oracleUseVelocityIncrement
        << " gridRes=" << oracleGridRes
        << " outputWeightFactor=" << oracleOutputWeightFactor
        << " stateReconstructionCorrectors="
        << oracleStateReconstructionCorrectors
        << " adaptiveStateReconstruction="
        << oracleAdaptiveStateReconstruction
        << " minStateReconstructionCorrectors="
        << oracleMinStateReconstructionCorrectors
        << " stateReconstructionResidualTolerance="
        << oracleStateReconstructionResidualTolerance
        << " stateReconstructionEquationRelaxation="
        << oracleStateReconstructionEquationRelaxation
        << " aitkenStateReconstruction="
        << oracleAitkenStateReconstruction
        << " aitkenFactorBounds=(" << oracleAitkenMinFactor
        << ' ' << oracleAitkenMaxFactor << ')'
        << " outerUResidualTolerance=" << outerUResidualTolerance
        << " outerPResidualTolerance=" << outerPResidualTolerance
        << " adaptivePressureCorrectors=" << adaptivePressureCorrectors
        << " minPressureCorrectors=" << minPressureCorrectors
        << " pressureCorrectorTolerance=" << pressureCorrectorTolerance
        << nl;

    // Full-state snapshots (deep field copies, never read/written to disk).
    // "Pre" = state at the start of the current timestep (both passes start
    // here). "Post" = pass 1's converged result (what the run actually keeps).
    volScalarField pPre(IOobject("oracleTwinPPre", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE), p);
    volVectorField UPre(IOobject("oracleTwinUPre", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE), U);
    surfaceScalarField phiPre(IOobject("oracleTwinPhiPre", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE), phi);
    volScalarField kPre(IOobject("oracleTwinKPre", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE), oracleTwinK);
    volScalarField omegaPre(IOobject("oracleTwinOmegaPre", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE), oracleTwinOmega);
    volScalarField nutPre(IOobject("oracleTwinNutPre", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE), oracleTwinNut);

    volScalarField pPost(IOobject("oracleTwinPPost", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE), p);
    volVectorField UPost(IOobject("oracleTwinUPost", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE), U);
    surfaceScalarField phiPost(IOobject("oracleTwinPhiPost", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE), phi);
    volScalarField kPost(IOobject("oracleTwinKPost", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE), oracleTwinK);
    volScalarField omegaPost(IOobject("oracleTwinOmegaPost", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE), oracleTwinOmega);
    volScalarField nutPost(IOobject("oracleTwinNutPost", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE), oracleTwinNut);

    // Per-pass GAMG diagnostic accumulators (referenced by name from pEqn.H).
    label oracleTwinPassTotalPIterations = 0;
    bool oracleTwinPassFirstResidualCaptured = false;
    scalar oracleTwinPassFirstResidual = 0.0;
    label oracleTwinPassFirstIterations = 0;
    bool oracleTwinReconstructionSolve = false;
    bool oracleTwinSkipMomentumSolve = false;

    scalar cumulativeReferenceWallTime = 0.0;
    scalar cumulativeOracleWallTime = 0.0;

    // Detailed cumulative wall-clock timings
    scalar cumulativeReferenceUEqnWallTime = 0.0;
    scalar cumulativeReferencePEqnWallTime = 0.0;
    scalar cumulativeReferenceKOmegaWallTime = 0.0;

    scalar cumulativeOracleUEqnWallTime = 0.0;
    scalar cumulativeOraclePEqnWallTime = 0.0;
    scalar cumulativeOracleKOmegaWallTime = 0.0;

    Info<< "\nStarting time loop\n" << endl;

    while (runTime.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        #include "CourantNo.H"
        #include "setDeltaT.H"

        // Start-of-timestep history capture, shared by both passes.
        p_prevTime = p;

        // Snapshot PRE state so pass 2 can be rewound to the same start.
        pPre = p; UPre = U; phiPre = phi;
        kPre = oracleTwinK; omegaPre = oracleTwinOmega; nutPre = oracleTwinNut;

        // ================= PASS 1: REFERENCE (truth) =================
        oracleTwinPassTotalPIterations = 0;
        oracleTwinPassFirstResidualCaptured = false;
        oracleTwinPassFirstResidual = 0.0;
        oracleTwinPassFirstIterations = 0;
        clearOracleTwinResiduals<scalar>(mesh);
        clearOracleTwinResiduals<vector>(mesh);

    scalar referenceUEqnWallTime = 0.0;
    scalar referencePEqnWallTime = 0.0;
    scalar referenceKOmegaWallTime = 0.0;

    const std::chrono::steady_clock::time_point referenceWallStart =
        std::chrono::steady_clock::now();

    {
        pimpleControl pimple(mesh);

        while (pimple.loop())
        {
            const label firstUSolve =
                oracleTwinResidualCount<vector>(mesh, U.name());

            // -------------------------------------------------------------
            // Momentum equation timing
            // -------------------------------------------------------------
            const std::chrono::steady_clock::time_point uEqnWallStart =
                std::chrono::steady_clock::now();

            #include "UEqn.H"

            referenceUEqnWallTime += std::chrono::duration<scalar>
            (
                std::chrono::steady_clock::now() - uEqnWallStart
            ).count();

            // -------------------------------------------------------------
            // Pressure corrections timing
            // -------------------------------------------------------------
            label referencePressureCorrectorsUsed = 0;
            scalar referenceLastCorrectorResidual = GREAT;

            while (pimple.correct())
            {
                ++referencePressureCorrectorsUsed;

                const label firstPressureSolveThisCorrector =
                    oracleTwinResidualCount<scalar>(mesh, p.name());

                const std::chrono::steady_clock::time_point pEqnWallStart =
                    std::chrono::steady_clock::now();

                #include "pEqn.H"

                referencePEqnWallTime += std::chrono::duration<scalar>
                (
                    std::chrono::steady_clock::now() - pEqnWallStart
                ).count();

                referenceLastCorrectorResidual =
                    oracleTwinFirstResidualSince<scalar>
                    (
                        mesh,
                        p.name(),
                        firstPressureSolveThisCorrector
                    );

                Info<< "[REFERENCE PISO] corrector="
                    << referencePressureCorrectorsUsed
                    << " firstPResidual="
                    << referenceLastCorrectorResidual
                    << " tolerance=" << pressureCorrectorTolerance
                    << nl;

                if
                (
                    adaptivePressureCorrectors
                 && referencePressureCorrectorsUsed
                    >= minPressureCorrectors
                 && referenceLastCorrectorResidual
                    < pressureCorrectorTolerance
                )
                {
                    Info<< "[REFERENCE PISO] stopping after "
                        << referencePressureCorrectorsUsed
                        << " pressure correctors" << nl;
                    break;
                }
            }

            // -------------------------------------------------------------
            // k-omega turbulence timing
            // -------------------------------------------------------------
            bool turbulenceCorrected = false;

            if (pimple.turbCorr())
            {
                const std::chrono::steady_clock::time_point
                    kOmegaWallStart =
                        std::chrono::steady_clock::now();

                laminarTransport.correct();
                turbulence->correct();

                referenceKOmegaWallTime += std::chrono::duration<scalar>
                (
                    std::chrono::steady_clock::now() - kOmegaWallStart
                ).count();

                turbulenceCorrected = true;
            }

            const scalar currentUResidual =
                oracleTwinFirstResidualSince<vector>
                (
                    mesh,
                    U.name(),
                    firstUSolve
                );

            const scalar currentPResidual =
                referenceLastCorrectorResidual;

            if
            (
                currentUResidual < outerUResidualTolerance
            && currentPResidual < outerPResidualTolerance
            )
            {
                // Ensure turbulence is corrected once before leaving PIMPLE.
                if (!turbulenceCorrected)
                {
                    const std::chrono::steady_clock::time_point
                        kOmegaWallStart =
                            std::chrono::steady_clock::now();

                    laminarTransport.correct();
                    turbulence->correct();

                    referenceKOmegaWallTime +=
                        std::chrono::duration<scalar>
                        (
                            std::chrono::steady_clock::now()
                        - kOmegaWallStart
                        ).count();
                }

                Info<< "[REFERENCE] outer residual tolerance satisfied: "
                    << "U=" << currentUResidual
                    << " p=" << currentPResidual
                    << " UTol=" << outerUResidualTolerance
                    << " pTol=" << outerPResidualTolerance
                    << nl;

                break;
            }
        }
    }

    scalar referenceWallTime = std::chrono::duration<scalar>
    (
        std::chrono::steady_clock::now() - referenceWallStart
    ).count();

    // The slowest MPI rank determines the elapsed wall time.
    reduce(referenceWallTime, maxOp<scalar>());
    reduce(referenceUEqnWallTime, maxOp<scalar>());
    reduce(referencePEqnWallTime, maxOp<scalar>());
    reduce(referenceKOmegaWallTime, maxOp<scalar>());


        const label refTotalPIter = oracleTwinPassTotalPIterations;
        const scalar refFirstResidual = oracleTwinPassFirstResidual;
        const label refFirstIter = oracleTwinPassFirstIterations;

        // Capture the sole reference-pass input permitted in the oracle pass.
        // All other oracle state is restored from the start of the timestep.
        pOracleGuess = p;

        // Pressure-only diagnostic fields; these do not carry any U/phi or
        // turbulence information into the oracle pass.
        dpTruth = p - p_prevTime;
        ddpTruth = dpTruth - dpPrevTime;

        // Snapshot the real advancing trajectory before rewinding for pass 2.
        pPost = p; UPost = U; phiPost = phi;
        kPost = oracleTwinK; omegaPost = oracleTwinOmega; nutPost = oracleTwinNut;

        // Perfect dU oracle from the same reference timestep. Apply the same
        // optional coarse-grid round trip and blend factor as for dp.
        oracleVelocityIncrement = UPost - UPre;

        if (oracleUseInterpolation)
        {
            oracleVelocityIncrement = oracleVectorGridRoundTrip
            (
                oracleVelocityIncrement,
                oracleGridRes
            );
        }

        oracleVelocityIncrement *= oracleOutputWeightFactor;

        // Rewind to the start of this timestep for the oracle pass.
        p = pPre; U = UPre; phi = phiPre;
        oracleTwinK = kPre; oracleTwinOmega = omegaPre; oracleTwinNut = nutPre;

        // The reference pass shares the mesh registry, where OpenFOAM stores
        // solver-performance records used by residualControl. Those records
        // are not field state and must not influence the oracle pass.
        clearOracleTwinResiduals<scalar>(mesh);
        clearOracleTwinResiduals<vector>(mesh);

        const std::chrono::steady_clock::time_point oracleWallStart =
            std::chrono::steady_clock::now();

        oraclePressureIncrement = pOracleGuess - p_prevTime;

        if (oracleUseInterpolation)
        {
            oraclePressureIncrement = oracleGridRoundTrip
            (
                oraclePressureIncrement,
                oracleGridRes
            );
        }

        oraclePressureIncrement *= oracleOutputWeightFactor;
        pOracleGuess = p_prevTime + oraclePressureIncrement;

        Info<< "[ORACLE TWIN] reconstruction pressure input="
            << (oracleUseInterpolation ? "interpolatedDp" : "fullResolutionDp")
            << " weightedDpMax="
            << gMax(mag(oraclePressureIncrement.internalField().field()))
            << nl;

        if (oracleUseVelocityIncrement)
        {
            U = UPre + oracleVelocityIncrement;
            U.correctBoundaryConditions();
            fvOptions.correct(U);

            phi = fvc::flux(U);
            MRF.makeRelative(phi);

            if (p.needReference())
            {
                fvc::makeRelative(phi, U);
                adjustPhi(phi, U, p);
                fvc::makeAbsolute(phi, U);
            }

            Info<< "[ORACLE TWIN] velocity oracle enabled"
                << " weightedDUmax="
                << gMax
                   (
                       mag
                       (
                           oracleVelocityIncrement.internalField().field()
                       )
                   )
                << nl;
        }
        else
        {
            Info<< "[ORACLE TWIN] pressure-only oracle enabled" << nl;
        }

        // ================= PASS 2: ORACLE (discarded, diagnostic only) ====
        oracleTwinPassTotalPIterations = 0;
        oracleTwinPassFirstResidualCaptured = false;
        oracleTwinPassFirstResidual = 0.0;
        oracleTwinPassFirstIterations = 0;
        scalar reconstructionWallTime = 0.0;
        scalar oraclePimpleWallTime = 0.0;

        scalar oracleUEqnWallTime = 0.0;
        scalar oraclePEqnWallTime = 0.0;
        scalar oracleKOmegaWallTime = 0.0;
        {
            bool oracleGuessInjected = false;
            pimpleControl pimple(mesh);

            if
            (
                (
                    oracleIntegrationMode == "rerunUEqnWithOraclePressure"
                 || oracleIntegrationMode == "pressureForceThenReconstruct"
                )
             && oracleStateReconstructionCorrectors > 0
            )
            {
                const std::chrono::steady_clock::time_point
                    reconstructionWallStart =
                        std::chrono::steady_clock::now();

                if
                (
                    oracleIntegrationMode == "pressureForceThenReconstruct"
                )
                {
                    // Assemble the relaxed momentum equation at the
                    // timestep-start state, but avoid a linear U solve.
                    oracleTwinSkipMomentumSolve = true;
                    #include "UEqn.H"
                    oracleTwinSkipMomentumSolve = false;

                    p = pOracleGuess;

                    const volScalarField rAUForce(1.0/UEqn.A());
                    const volVectorField HbyAForce
                    (
                        constrainHbyA(rAUForce*UEqn.H(), U, p)
                    );
                    surfaceScalarField phiHbyAForce
                    (
                        "phiHbyAForce",
                        fvc::flux(HbyAForce)
                      + MRF.zeroFilter
                        (
                            fvc::interpolate(rAUForce)*fvc::ddtCorr(U, phi)
                        )
                    );

                    MRF.makeRelative(phiHbyAForce);

                    if (p.needReference())
                    {
                        fvc::makeRelative(phiHbyAForce, U);
                        adjustPhi(phiHbyAForce, U, p);
                        fvc::makeAbsolute(phiHbyAForce, U);
                    }

                    constrainPressure
                    (
                        p,
                        U,
                        phiHbyAForce,
                        rAUForce,
                        MRF
                    );

                    fvScalarMatrix pEqnForce
                    (
                        fvm::laplacian(rAUForce, p)
                     == fvc::div(phiHbyAForce)
                    );
                    pEqnForce.setReference(pRefCell, pRefValue);

                    phi = phiHbyAForce - pEqnForce.flux();
                    U = HbyAForce - rAUForce*fvc::grad(p);
                    U.correctBoundaryConditions();
                    fvOptions.correct(U);

                    Info<< "[ORACLE TWIN] applied no-solve HbyA pressure "
                        << "predictor before reconstruction" << nl;
                }
                else
                {
                    p = pOracleGuess;
                }

                oracleGuessInjected = true;
                label reconstructionCorrectorsUsed = 0;
                scalar reconstructionResidual = GREAT;
                scalar aitkenFactor = 1.1;
                bool havePreviousReconstructionDelta = false;
                volVectorField reconstructionUBefore
                (
                    IOobject
                    (
                        "oracleReconstructionUBefore",
                        runTime.timeName(),
                        mesh,
                        IOobject::NO_READ,
                        IOobject::NO_WRITE
                    ),
                    U
                );
                surfaceScalarField reconstructionPhiBefore
                (
                    IOobject
                    (
                        "oracleReconstructionPhiBefore",
                        runTime.timeName(),
                        mesh,
                        IOobject::NO_READ,
                        IOobject::NO_WRITE
                    ),
                    phi
                );
                volVectorField previousReconstructionDelta
                (
                    IOobject
                    (
                        "oraclePreviousReconstructionDelta",
                        runTime.timeName(),
                        mesh,
                        IOobject::NO_READ,
                        IOobject::NO_WRITE
                    ),
                    U - U
                );

                for
                (
                    label reconstructionCorr = 0;
                    reconstructionCorr < oracleStateReconstructionCorrectors;
                    ++reconstructionCorr
                )
                {
                    // Reassert the only reference-pass information before
                    // deriving this corrector's U/phi/turbulence state.
                    p = pOracleGuess;
                    reconstructionUBefore = U;
                    reconstructionPhiBefore = phi;

                    const label firstReconstructionUSolve =
                        oracleTwinResidualCount<vector>(mesh, U.name());

                    oracleTwinReconstructionSolve = true;
                    #include "UEqn.H"
                    oracleTwinReconstructionSolve = false;

                    reconstructionResidual =
                        oracleTwinFirstResidualSince<vector>
                        (
                            mesh,
                            U.name(),
                            firstReconstructionUSolve
                        );
                    reconstructionCorrectorsUsed = reconstructionCorr + 1;

                    volScalarField rAUReconstruct(1.0/UEqn.A());
                    volVectorField HbyAReconstruct
                    (
                        constrainHbyA(rAUReconstruct*UEqn.H(), U, p)
                    );
                    surfaceScalarField phiHbyAReconstruct
                    (
                        "phiHbyAReconstruct",
                        fvc::flux(HbyAReconstruct)
                      + MRF.zeroFilter
                        (
                            fvc::interpolate(rAUReconstruct)
                           *fvc::ddtCorr(U, phi)
                        )
                    );

                    MRF.makeRelative(phiHbyAReconstruct);

                    if (p.needReference())
                    {
                        fvc::makeRelative(phiHbyAReconstruct, U);
                        adjustPhi(phiHbyAReconstruct, U, p);
                        fvc::makeAbsolute(phiHbyAReconstruct, U);
                    }

                    constrainPressure
                    (
                        p,
                        U,
                        phiHbyAReconstruct,
                        rAUReconstruct,
                        MRF
                    );

                    fvScalarMatrix pEqnReconstruct
                    (
                        fvm::laplacian(rAUReconstruct, p)
                     == fvc::div(phiHbyAReconstruct)
                    );
                    pEqnReconstruct.setReference(pRefCell, pRefValue);

                    phi = phiHbyAReconstruct - pEqnReconstruct.flux();
                    U = HbyAReconstruct - rAUReconstruct*fvc::grad(p);
                    U.correctBoundaryConditions();
                    fvOptions.correct(U);

                    volVectorField reconstructionDelta
                    (
                        IOobject
                        (
                            "oracleReconstructionDelta",
                            runTime.timeName(),
                            mesh,
                            IOobject::NO_READ,
                            IOobject::NO_WRITE,
                            false
                        ),
                        U - reconstructionUBefore
                    );

                    if
                    (
                        oracleAitkenStateReconstruction
                     && havePreviousReconstructionDelta
                    )
                    {
                        const volVectorField deltaChange
                        (
                            IOobject
                            (
                                "oracleReconstructionDeltaChange",
                                runTime.timeName(),
                                mesh,
                                IOobject::NO_READ,
                                IOobject::NO_WRITE,
                                false
                            ),
                            reconstructionDelta
                          - previousReconstructionDelta
                        );

                        scalar numerator = 0.0;
                        scalar denominator = 0.0;
                        forAll(U, celli)
                        {
                            numerator += mesh.V()[celli]
                                *(previousReconstructionDelta[celli]
                                & deltaChange[celli]);
                            denominator += mesh.V()[celli]
                                *magSqr(deltaChange[celli]);
                        }
                        reduce(numerator, sumOp<scalar>());
                        reduce(denominator, sumOp<scalar>());

                        if (denominator > VSMALL)
                        {
                            aitkenFactor = min
                            (
                                oracleAitkenMaxFactor,
                                max
                                (
                                    oracleAitkenMinFactor,
                                   -aitkenFactor*numerator/denominator
                                )
                            );
                        }

                        U = reconstructionUBefore
                          + aitkenFactor*reconstructionDelta;
                        phi = reconstructionPhiBefore
                          + aitkenFactor
                           *(phi - reconstructionPhiBefore);
                        U.correctBoundaryConditions();
                        fvOptions.correct(U);
                    }

                    previousReconstructionDelta = reconstructionDelta;
                    havePreviousReconstructionDelta = true;

                    Info<< "[ORACLE TWIN] reconstruction="
                        << reconstructionCorrectorsUsed
                        << " UResidual=" << reconstructionResidual
                        << " aitkenFactor=" << aitkenFactor << nl;

                    if
                    (
                        oracleAdaptiveStateReconstruction
                     && reconstructionCorrectorsUsed
                        >= oracleMinStateReconstructionCorrectors
                     && reconstructionResidual
                        < oracleStateReconstructionResidualTolerance
                    )
                    {
                        break;
                    }
                }

                Info<< "[ORACLE TWIN] reconstructed pressure-derived state "
                    << "with " << reconstructionCorrectorsUsed
                    << " fixed-pressure correctors"
                    << " finalUResidual=" << reconstructionResidual << nl;

                                reconstructionWallTime = std::chrono::duration<scalar>
                                (
                                        std::chrono::steady_clock::now()
                                    - reconstructionWallStart
                                ).count();
            }

            // Exclude reconstruction U records so measured residual control
            // starts from the first oracle PIMPLE correction.
            clearOracleTwinResiduals<scalar>(mesh);
            clearOracleTwinResiduals<vector>(mesh);

            const std::chrono::steady_clock::time_point oraclePimpleWallStart =
                std::chrono::steady_clock::now();

            while (pimple.loop())
            {
                const label firstUSolve =
                    oracleTwinResidualCount<vector>(mesh, U.name());

                if
                (
                    pimple.firstPimpleIter()
                 &&
                    (
                        oracleIntegrationMode
                     == "rerunUEqnWithOraclePressure"
                     || oracleIntegrationMode
                     == "pressureForceThenReconstruct"
                    )
                )
                {
                    p = pOracleGuess;
                    oracleGuessInjected = true;
                }

                const std::chrono::steady_clock::time_point uEqnWallStart =
                    std::chrono::steady_clock::now();

                #include "UEqn.H"

                oracleUEqnWallTime += std::chrono::duration<scalar>
                (
                    std::chrono::steady_clock::now() - uEqnWallStart
                ).count();

                if (pimple.firstPimpleIter() && !oracleGuessInjected)
                {
                    const volScalarField dpSM(pOracleGuess - p_prevTime);
                    p = pOracleGuess;
                    oracleGuessInjected = true;

                    if
                    (
                        oracleIntegrationMode == "directPressureForceCorrection"
                    )
                    {
                        const volScalarField rAUcorr(1.0/UEqn.A());
                        U = U - rAUcorr*fvc::grad(dpSM);
                        U.correctBoundaryConditions();
                        fvOptions.correct(U);

                        phi = fvc::flux(U);
                        MRF.makeRelative(phi);
                        if (p.needReference())
                        {
                            fvc::makeRelative(phi, U);
                            adjustPhi(phi, U, p);
                            fvc::makeAbsolute(phi, U);
                        }
                    }
                }

                label oraclePressureCorrectorsUsed = 0;
                scalar oracleLastCorrectorResidual = GREAT;

                while (pimple.correct())
                {
                    ++oraclePressureCorrectorsUsed;

                    const label firstPressureSolveThisCorrector =
                        oracleTwinResidualCount<scalar>(mesh, p.name());

                    const std::chrono::steady_clock::time_point pEqnWallStart =
                        std::chrono::steady_clock::now();

                    #include "pEqn.H"

                    oraclePEqnWallTime += std::chrono::duration<scalar>
                    (
                        std::chrono::steady_clock::now() - pEqnWallStart
                    ).count();

                    oracleLastCorrectorResidual =
                        oracleTwinFirstResidualSince<scalar>
                        (
                            mesh,
                            p.name(),
                            firstPressureSolveThisCorrector
                        );

                    Info<< "[ORACLE PISO] corrector="
                        << oraclePressureCorrectorsUsed
                        << " firstPResidual="
                        << oracleLastCorrectorResidual
                        << " tolerance=" << pressureCorrectorTolerance
                        << nl;

                    if
                    (
                        adaptivePressureCorrectors
                     && oraclePressureCorrectorsUsed
                        >= minPressureCorrectors
                     && oracleLastCorrectorResidual
                        < pressureCorrectorTolerance
                    )
                    {
                        Info<< "[ORACLE PISO] stopping after "
                            << oraclePressureCorrectorsUsed
                            << " pressure correctors" << nl;
                        break;
                    }
                }

                bool turbulenceCorrected = false;

                if (pimple.turbCorr())
                {
                    const std::chrono::steady_clock::time_point kOmegaWallStart =
                        std::chrono::steady_clock::now();

                    laminarTransport.correct();
                    turbulence->correct();

                    oracleKOmegaWallTime += std::chrono::duration<scalar>
                    (
                        std::chrono::steady_clock::now() - kOmegaWallStart
                    ).count();

                    turbulenceCorrected = true;
                }

                const scalar currentUResidual =
                    oracleTwinFirstResidualSince<vector>
                    (
                        mesh,
                        U.name(),
                        firstUSolve
                    );
                const scalar currentPResidual =
                    oracleLastCorrectorResidual;

                if
                (
                          currentUResidual < outerUResidualTolerance
                      && currentPResidual < outerPResidualTolerance
                )
                {

                    if (!turbulenceCorrected)
                    {
                        const std::chrono::steady_clock::time_point kOmegaWallStart =
                            std::chrono::steady_clock::now();

                        laminarTransport.correct();
                        turbulence->correct();

                        oracleKOmegaWallTime += std::chrono::duration<scalar>
                        (
                            std::chrono::steady_clock::now() - kOmegaWallStart
                        ).count();
                    }
                  

                    Info<< "[ORACLE TWIN] outer residual tolerance satisfied: "
                        << "U=" << currentUResidual
                        << " p=" << currentPResidual
                        << " UTol=" << outerUResidualTolerance
                        << " pTol=" << outerPResidualTolerance
                        << nl;
                    break;
                }
            }

            oraclePimpleWallTime = std::chrono::duration<scalar>
            (
                std::chrono::steady_clock::now() - oraclePimpleWallStart
            ).count();
        }
        scalar oracleWallTime = std::chrono::duration<scalar>
        (
            std::chrono::steady_clock::now() - oracleWallStart
        ).count();

        reduce(reconstructionWallTime, maxOp<scalar>());
        reduce(oraclePimpleWallTime, maxOp<scalar>());
        reduce(oracleWallTime, maxOp<scalar>());

        reduce(oracleUEqnWallTime, maxOp<scalar>());
        reduce(oraclePEqnWallTime, maxOp<scalar>());
        reduce(oracleKOmegaWallTime, maxOp<scalar>());

        const label oracleTotalPIter = oracleTwinPassTotalPIterations;
        const scalar oracleFirstResidual = oracleTwinPassFirstResidual;
        const label oracleFirstIter = oracleTwinPassFirstIterations;

        Info<< "[ORACLE TWIN] time=" << runTime.timeName()
            << " reference: firstResid=" << refFirstResidual
            << " firstIter=" << refFirstIter
            << " totalIter=" << refTotalPIter
            << " | oracle: firstResid=" << oracleFirstResidual
            << " firstIter=" << oracleFirstIter
            << " totalIter=" << oracleTotalPIter
            << " | totalIter reduction=" << (refTotalPIter - oracleTotalPIter)
            << nl << endl;

        cumulativeReferenceWallTime += referenceWallTime;
        cumulativeOracleWallTime += oracleWallTime;

        cumulativeReferenceUEqnWallTime += referenceUEqnWallTime;
        cumulativeReferencePEqnWallTime += referencePEqnWallTime;
        cumulativeReferenceKOmegaWallTime += referenceKOmegaWallTime;
        cumulativeOracleUEqnWallTime += oracleUEqnWallTime;
        cumulativeOraclePEqnWallTime += oraclePEqnWallTime;
        cumulativeOracleKOmegaWallTime += oracleKOmegaWallTime;

        Info<< "[ORACLE COST] time=" << runTime.timeName()
            << " referenceWall=" << referenceWallTime << " s"
            << " oracleReconstructionWall=" << reconstructionWallTime << " s"
            << " oraclePimpleWall=" << oraclePimpleWallTime << " s"
            << " oracleTotalWall=" << oracleWallTime << " s"
            << " stepSpeedup="
            << referenceWallTime/max(oracleWallTime, SMALL)
            << " cumulativeSpeedup="
            << cumulativeReferenceWallTime
               /max(cumulativeOracleWallTime, SMALL)
            << nl << endl;


            // ---------------------------------------------------------------------
            // Detailed equation timing report
            // ---------------------------------------------------------------------

            const scalar referenceMeasuredWallTime =
                referenceUEqnWallTime
            + referencePEqnWallTime
            + referenceKOmegaWallTime;

            const scalar oracleMeasuredWallTime =
                oracleUEqnWallTime
            + oraclePEqnWallTime
            + oracleKOmegaWallTime;

            const scalar referenceOtherWallTime = max
            (
                scalar(0),
                referenceWallTime - referenceMeasuredWallTime
            );

            const scalar oraclePimpleOtherWallTime = max
            (
                scalar(0),
                oraclePimpleWallTime - oracleMeasuredWallTime
            );

            Info<< "[REFERENCE EQUATION COST]"
                << " time=" << runTime.timeName()
                << " UEqn=" << referenceUEqnWallTime << " s"
                << " pEqns=" << referencePEqnWallTime << " s"
                << " kOmega=" << referenceKOmegaWallTime << " s"
                << " other=" << referenceOtherWallTime << " s"
                << " total=" << referenceWallTime << " s"
                << nl;

            Info<< "[REFERENCE EQUATION FRACTIONS]"
                << " time=" << runTime.timeName()
                << " UEqn="
                << referenceUEqnWallTime/max(referenceWallTime, SMALL)
                << " pEqns="
                << referencePEqnWallTime/max(referenceWallTime, SMALL)
                << " kOmega="
                << referenceKOmegaWallTime/max(referenceWallTime, SMALL)
                << " other="
                << referenceOtherWallTime/max(referenceWallTime, SMALL)
                << nl;

            Info<< "[ORACLE EQUATION COST]"
                << " time=" << runTime.timeName()
                << " UEqn=" << oracleUEqnWallTime << " s"
                << " pEqns=" << oraclePEqnWallTime << " s"
                << " kOmega=" << oracleKOmegaWallTime << " s"
                << " pimpleOther=" << oraclePimpleOtherWallTime << " s"
                << " reconstruction=" << reconstructionWallTime << " s"
                << " pimpleTotal=" << oraclePimpleWallTime << " s"
                << " oracleTotal=" << oracleWallTime << " s"
                << nl;

            Info<< "[ORACLE EQUATION FRACTIONS]"
                << " time=" << runTime.timeName()
                << " UEqn="
                << oracleUEqnWallTime/max(oraclePimpleWallTime, SMALL)
                << " pEqns="
                << oraclePEqnWallTime/max(oraclePimpleWallTime, SMALL)
                << " kOmega="
                << oracleKOmegaWallTime/max(oraclePimpleWallTime, SMALL)
                << " pimpleOther="
                << oraclePimpleOtherWallTime/max(oraclePimpleWallTime, SMALL)
                << nl << endl;

        Info<< "[CUMULATIVE EQUATION COST]"
            << " referenceUEqn=" << cumulativeReferenceUEqnWallTime << " s"
            << " referencePEqns=" << cumulativeReferencePEqnWallTime << " s"
            << " referenceKOmega=" << cumulativeReferenceKOmegaWallTime << " s"
            << " oracleUEqn=" << cumulativeOracleUEqnWallTime << " s"
            << " oraclePEqns=" << cumulativeOraclePEqnWallTime << " s"
            << " oracleKOmega=" << cumulativeOracleKOmegaWallTime << " s"
            << nl << endl;

        // Restore the real (reference) trajectory — pass 2 is diagnostic
        // only and must not influence the simulation going forward.
        p = pPost; U = UPost; phi = phiPost;
        oracleTwinK = kPost; oracleTwinOmega = omegaPost; oracleTwinNut = nutPost;

        // Finalize history for the next timestep.
        dpPrevTime = dpTruth;
        ddpPrevTime = ddpTruth;

        if (runTime.writeTime())
        {
            oraclePressureIncrement.write();
            oracleVelocityIncrement.write();
            pOracleGuess.write();
        }

        runTime.write();

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    Info<< "End\n" << endl;

    return 0;
}
