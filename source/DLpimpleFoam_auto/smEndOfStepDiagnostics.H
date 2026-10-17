#ifndef smEndOfStepDiagnostics_H
#define smEndOfStepDiagnostics_H

// End-of-timestep diagnostics for the surrogate pressure prediction.
//
// This fragment is included after the PIMPLE loop, when p contains the
// converged CFD pressure for the current timestep. The fields pExtrap and pSM
// remain frozen at their first-PIMPLE-iteration values.
//
// Comparisons are made against both simple baselines:
//   1. previous pressure only:       p_prev
//   2. linear temporal extrapolation: p_prev + dpPrev
//
// Positive improvement means that pSM is closer to converged p than the
// corresponding baseline. All RMS values are volume weighted.

if (smDiagnosticsActiveThisStep)
{
    const volScalarField smDiagPreviousOnly
    (
        IOobject
        (
            "smDiagPreviousOnly",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false
        ),
        p_prev
    );

    const volScalarField smDiagTrend
    (
        IOobject
        (
            "smDiagTrend",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false
        ),
        p_prev + dpPrev
    );

    // pExtrap should be identical to one of the two baselines depending on
    // prediction mode. Keep a dedicated comparison for continuity with the
    // original diagnostics and its cumulative counters.
    const Tuple2<scalar, scalar> pressureExtrapMetrics =
        computeScalarErrorMetrics(pExtrap, p);

    const Tuple2<scalar, scalar> pressurePreviousMetrics =
        computeScalarErrorMetrics(smDiagPreviousOnly, p);

    const Tuple2<scalar, scalar> pressureTrendMetrics =
        computeScalarErrorMetrics(smDiagTrend, p);

    const Tuple2<scalar, scalar> pressureSMMetrics =
        computeScalarErrorMetrics(pSM, p);

    // Pressure-gradient comparisons.
    const volVectorField smDiagGradTruth
    (
        IOobject
        (
            "smDiagGradTruth",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false
        ),
        fvc::grad(p)
    );

    const volVectorField smDiagGradExtrap
    (
        IOobject
        (
            "smDiagGradExtrap",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false
        ),
        fvc::grad(pExtrap)
    );

    const volVectorField smDiagGradPrevious
    (
        IOobject
        (
            "smDiagGradPrevious",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false
        ),
        fvc::grad(smDiagPreviousOnly)
    );

    const volVectorField smDiagGradTrend
    (
        IOobject
        (
            "smDiagGradTrend",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false
        ),
        fvc::grad(smDiagTrend)
    );

    const volVectorField smDiagGradSM
    (
        IOobject
        (
            "smDiagGradSM",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false
        ),
        fvc::grad(pSM)
    );

    const Tuple2<scalar, scalar> gradientExtrapMetrics =
        computeVectorErrorMagnitudeMetrics
        (
            smDiagGradExtrap,
            smDiagGradTruth
        );

    const Tuple2<scalar, scalar> gradientPreviousMetrics =
        computeVectorErrorMagnitudeMetrics
        (
            smDiagGradPrevious,
            smDiagGradTruth
        );

    const Tuple2<scalar, scalar> gradientTrendMetrics =
        computeVectorErrorMagnitudeMetrics
        (
            smDiagGradTrend,
            smDiagGradTruth
        );

    const Tuple2<scalar, scalar> gradientSMMetrics =
        computeVectorErrorMagnitudeMetrics
        (
            smDiagGradSM,
            smDiagGradTruth
        );

    // Unweighted Laplacian comparisons. These diagnose mesh-scale pressure
    // structure independently of the rAU-weighted pressure operator.
    const volScalarField smDiagLaplacianTruth
    (
        IOobject
        (
            "smDiagLaplacianTruth",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false
        ),
        fvc::laplacian(p)
    );

    const volScalarField smDiagLaplacianExtrap
    (
        IOobject
        (
            "smDiagLaplacianExtrap",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false
        ),
        fvc::laplacian(pExtrap)
    );

    const volScalarField smDiagLaplacianPrevious
    (
        IOobject
        (
            "smDiagLaplacianPrevious",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false
        ),
        fvc::laplacian(smDiagPreviousOnly)
    );

    const volScalarField smDiagLaplacianTrend
    (
        IOobject
        (
            "smDiagLaplacianTrend",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false
        ),
        fvc::laplacian(smDiagTrend)
    );

    const volScalarField smDiagLaplacianSM
    (
        IOobject
        (
            "smDiagLaplacianSM",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false
        ),
        fvc::laplacian(pSM)
    );

    const Tuple2<scalar, scalar> laplacianExtrapMetrics =
        computeScalarErrorMetrics
        (
            smDiagLaplacianExtrap,
            smDiagLaplacianTruth
        );

    const Tuple2<scalar, scalar> laplacianPreviousMetrics =
        computeScalarErrorMetrics
        (
            smDiagLaplacianPrevious,
            smDiagLaplacianTruth
        );

    const Tuple2<scalar, scalar> laplacianTrendMetrics =
        computeScalarErrorMetrics
        (
            smDiagLaplacianTrend,
            smDiagLaplacianTruth
        );

    const Tuple2<scalar, scalar> laplacianSMMetrics =
        computeScalarErrorMetrics
        (
            smDiagLaplacianSM,
            smDiagLaplacianTruth
        );

    // Per-step improvements relative to the active pExtrap baseline.
    const scalar pressureImprovement = computeImprovementPercent
    (
        pressureExtrapMetrics.first(),
        pressureSMMetrics.first()
    );

    const scalar gradientImprovement = computeImprovementPercent
    (
        gradientExtrapMetrics.first(),
        gradientSMMetrics.first()
    );

    const scalar laplacianImprovement = computeImprovementPercent
    (
        laplacianExtrapMetrics.first(),
        laplacianSMMetrics.first()
    );

    // Per-step improvements relative to both mode-independent baselines.
    const scalar pressureImprovementVsPrevious = computeImprovementPercent
    (
        pressurePreviousMetrics.first(),
        pressureSMMetrics.first()
    );

    const scalar pressureImprovementVsTrend = computeImprovementPercent
    (
        pressureTrendMetrics.first(),
        pressureSMMetrics.first()
    );

    const scalar gradientImprovementVsPrevious = computeImprovementPercent
    (
        gradientPreviousMetrics.first(),
        gradientSMMetrics.first()
    );

    const scalar gradientImprovementVsTrend = computeImprovementPercent
    (
        gradientTrendMetrics.first(),
        gradientSMMetrics.first()
    );

    const scalar laplacianImprovementVsPrevious = computeImprovementPercent
    (
        laplacianPreviousMetrics.first(),
        laplacianSMMetrics.first()
    );

    const scalar laplacianImprovementVsTrend = computeImprovementPercent
    (
        laplacianTrendMetrics.first(),
        laplacianSMMetrics.first()
    );

    ++smDiagnosticsSamples;

    cumulativePressureRMSImprovement += pressureImprovement;
    cumulativeGradientRMSImprovement += gradientImprovement;
    cumulativeLaplacianRMSImprovement += laplacianImprovement;

    cumulativePressureRMSImprovementVsPrevOnly +=
        pressureImprovementVsPrevious;
    cumulativePressureRMSImprovementVsTrend +=
        pressureImprovementVsTrend;

    cumulativeGradientRMSImprovementVsPrevOnly +=
        gradientImprovementVsPrevious;
    cumulativeGradientRMSImprovementVsTrend +=
        gradientImprovementVsTrend;

    cumulativeLaplacianRMSImprovementVsPrevOnly +=
        laplacianImprovementVsPrevious;
    cumulativeLaplacianRMSImprovementVsTrend +=
        laplacianImprovementVsTrend;

    // Operator and continuity diagnostics are captured in pEqn.H because the
    // corresponding rAU and phiHbyA fields exist only inside that include.
    // Add their per-step values here only when pEqn.H populated them.
    if (smOperatorDiagnosticsDoneThisStep)
    {
        cumulativeOperatorRMSImprovement +=
            smOperatorRMSImprovementThisStep;
        cumulativeContinuityRMSImprovement +=
            smContinuityRMSImprovementThisStep;
    }

    Info<< nl
        << "[SM DIAGNOSTICS] ========================================" << nl
        << "[SM DIAGNOSTICS] End-of-step comparison at Time="
        << runTime.timeName() << nl;

    printMetricAssessment3
    (
        "Pressure",
        pressurePreviousMetrics.first(),
        pressureTrendMetrics.first(),
        pressureSMMetrics.first()
    );

    printMetricAssessment3
    (
        "Gradient",
        gradientPreviousMetrics.first(),
        gradientTrendMetrics.first(),
        gradientSMMetrics.first()
    );

    printMetricAssessment3
    (
        "Laplacian",
        laplacianPreviousMetrics.first(),
        laplacianTrendMetrics.first(),
        laplacianSMMetrics.first()
    );

    Info<< "[SM DIAGNOSTICS] Max absolute errors:"
        << " pressure=" << pressureSMMetrics.second()
        << " gradient=" << gradientSMMetrics.second()
        << " laplacian=" << laplacianSMMetrics.second()
        << nl;

    if (smOperatorDiagnosticsDoneThisStep)
    {
        Info<< "[SM DIAGNOSTICS] Pressure operator: extrap RMS="
            << smOperatorRmsExtrapThisStep
            << " | SM RMS=" << smOperatorRmsSMThisStep
            << " | improvement="
            << signedStr(smOperatorRMSImprovementThisStep) << '%'
            << " (" << helpedState(smOperatorRMSImprovementThisStep)
            << ')' << nl;

        Info<< "[SM DIAGNOSTICS] Continuity improvement="
            << signedStr(smContinuityRMSImprovementThisStep) << '%'
            << " (" << helpedState(smContinuityRMSImprovementThisStep)
            << ')' << nl;
    }
    else
    {
        Info<< "[SM DIAGNOSTICS] Operator/continuity comparison was not "
            << "captured in pEqn.H for this timestep." << nl;
    }

    if (smFirstGamgResidualCapturedThisStep)
    {
        Info<< "[SM DIAGNOSTICS] First GAMG initial residual="
            << smFirstGamgInitialResidualThisStep << nl;
    }

    const scalar sampleCount = scalar(smDiagnosticsSamples);

    Info<< "[SM DIAGNOSTICS] Cumulative mean improvement vs active pExtrap:"
        << " pressure="
        << cumulativePressureRMSImprovement/sampleCount << '%'
        << " gradient="
        << cumulativeGradientRMSImprovement/sampleCount << '%'
        << " laplacian="
        << cumulativeLaplacianRMSImprovement/sampleCount << '%'
        << nl;

    Info<< "[SM DIAGNOSTICS] Cumulative mean improvement vs p_prev:"
        << " pressure="
        << cumulativePressureRMSImprovementVsPrevOnly/sampleCount << '%'
        << " gradient="
        << cumulativeGradientRMSImprovementVsPrevOnly/sampleCount << '%'
        << " laplacian="
        << cumulativeLaplacianRMSImprovementVsPrevOnly/sampleCount << '%'
        << nl;

    Info<< "[SM DIAGNOSTICS] Cumulative mean improvement vs trend:"
        << " pressure="
        << cumulativePressureRMSImprovementVsTrend/sampleCount << '%'
        << " gradient="
        << cumulativeGradientRMSImprovementVsTrend/sampleCount << '%'
        << " laplacian="
        << cumulativeLaplacianRMSImprovementVsTrend/sampleCount << '%'
        << nl;

    if (smOperatorDiagnosticsDoneThisStep)
    {
        Info<< "[SM DIAGNOSTICS] Cumulative operator/continuity sums "
            << "over diagnostic pressure-equation captures:"
            << " operator=" << cumulativeOperatorRMSImprovement << '%'
            << " continuity=" << cumulativeContinuityRMSImprovement << '%'
            << nl;
    }

    Info<< "[SM DIAGNOSTICS] ========================================"
        << nl << endl;
}

#endif
