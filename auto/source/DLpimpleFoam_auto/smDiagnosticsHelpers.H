#ifndef smDiagnosticsHelpers_H
#define smDiagnosticsHelpers_H

// Helper lambdas used by pEqn.H and smEndOfStepDiagnostics.H.
// This file is included inside main(), after the diagnostic state variables
// have been declared. The lambdas therefore deliberately capture by reference.

// Return value layout for scalar/vector metrics:
//   first()  = volume-weighted RMS error
//   second() = global maximum absolute error (or vector-error magnitude)
auto computeScalarErrorMetrics =
[&]
(
    const volScalarField& estimate,
    const volScalarField& reference
) -> Tuple2<scalar, scalar>
{
    scalar weightedErrorSquared = 0.0;
    scalar totalVolume = 0.0;
    scalar maximumAbsoluteError = 0.0;

    const scalarField& estimateInternal = estimate.internalField();
    const scalarField& referenceInternal = reference.internalField();
    const scalarField& cellVolumes = mesh.V();

    forAll(estimateInternal, celli)
    {
        const scalar error =
            estimateInternal[celli] - referenceInternal[celli];

        weightedErrorSquared += cellVolumes[celli]*sqr(error);
        totalVolume += cellVolumes[celli];
        maximumAbsoluteError = max(maximumAbsoluteError, mag(error));
    }

    reduce(weightedErrorSquared, sumOp<scalar>());
    reduce(totalVolume, sumOp<scalar>());
    reduce(maximumAbsoluteError, maxOp<scalar>());

    const scalar rmsError =
        Foam::sqrt(weightedErrorSquared/Foam::max(totalVolume, VSMALL));

    return Tuple2<scalar, scalar>(rmsError, maximumAbsoluteError);
};


auto computeVectorErrorMagnitudeMetrics =
[&]
(
    const volVectorField& estimate,
    const volVectorField& reference
) -> Tuple2<scalar, scalar>
{
    scalar weightedErrorSquared = 0.0;
    scalar totalVolume = 0.0;
    scalar maximumErrorMagnitude = 0.0;

    const vectorField& estimateInternal = estimate.internalField();
    const vectorField& referenceInternal = reference.internalField();
    const scalarField& cellVolumes = mesh.V();

    forAll(estimateInternal, celli)
    {
        const scalar errorMagnitude = mag
        (
            estimateInternal[celli] - referenceInternal[celli]
        );

        weightedErrorSquared +=
            cellVolumes[celli]*sqr(errorMagnitude);
        totalVolume += cellVolumes[celli];
        maximumErrorMagnitude = max
        (
            maximumErrorMagnitude,
            errorMagnitude
        );
    }

    reduce(weightedErrorSquared, sumOp<scalar>());
    reduce(totalVolume, sumOp<scalar>());
    reduce(maximumErrorMagnitude, maxOp<scalar>());

    const scalar rmsError =
        Foam::sqrt(weightedErrorSquared/Foam::max(totalVolume, VSMALL));

    return Tuple2<scalar, scalar>(rmsError, maximumErrorMagnitude);
};


// Positive means the SM reduced the error. Negative means it increased it.
auto computeImprovementPercent =
[]
(
    const scalar baselineError,
    const scalar surrogateError
) -> scalar
{
    if (baselineError <= VSMALL)
    {
        return
            surrogateError <= VSMALL
          ? scalar(0)
          : -GREAT;
    }

    return 100.0*(baselineError - surrogateError)/baselineError;
};


auto helpedState =
[]
(
    const scalar improvementPercent
) -> word
{
    const scalar neutralTolerance = 1e-10;

    if (improvementPercent > neutralTolerance)
    {
        return word("HELPED");
    }

    if (improvementPercent < -neutralTolerance)
    {
        return word("HURT");
    }

    return word("NEUTRAL");
};


auto signedStr =
[]
(
    const scalar value
) -> std::string
{
    std::ostringstream output;

    if (value >= 0.0)
    {
        output << '+';
    }

    output
        << std::fixed
        << std::setprecision(2)
        << value;

    return output.str();
};


// Print one comparison: a baseline against the SM.
auto printMetricAssessment =
[&]
(
    const word& metricName,
    const scalar baselineRMS,
    const scalar surrogateRMS
)
{
    const scalar improvement = computeImprovementPercent
    (
        baselineRMS,
        surrogateRMS
    );

    Info<< "[SM DIAGNOSTICS] " << metricName
        << ": baseline RMS=" << baselineRMS
        << " | SM RMS=" << surrogateRMS
        << " | improvement=" << signedStr(improvement) << '%'
        << " | " << helpedState(improvement)
        << nl;
};


// Print the SM against both naive pressure baselines:
//   1. p_prev
//   2. p_prev + dpPrev
// This is the three-way comparison described by the recovered main solver.
auto printMetricAssessment3 =
[&]
(
    const word& metricName,
    const scalar previousOnlyRMS,
    const scalar trendRMS,
    const scalar surrogateRMS
)
{
    const scalar improvementVsPrevious = computeImprovementPercent
    (
        previousOnlyRMS,
        surrogateRMS
    );

    const scalar improvementVsTrend = computeImprovementPercent
    (
        trendRMS,
        surrogateRMS
    );

    Info<< "[SM DIAGNOSTICS] " << metricName
        << ": p_prev only RMS=" << previousOnlyRMS
        << " | p_prev + dpPrev RMS=" << trendRMS
        << " | SM RMS=" << surrogateRMS
        << " | vs p_prev=" << signedStr(improvementVsPrevious) << '%'
        << " (" << helpedState(improvementVsPrevious) << ')'
        << " | vs trend=" << signedStr(improvementVsTrend) << '%'
        << " (" << helpedState(improvementVsTrend) << ')'
        << nl;
};

#endif
