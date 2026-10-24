volScalarField rAU(1.0/UEqn.A());

volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));

surfaceScalarField phiHbyA
(
    "phiHbyA",
    fvc::flux(HbyA)
  + MRF.zeroFilter(fvc::interpolate(rAU)*fvc::ddtCorr(U, phi))
);

MRF.makeRelative(phiHbyA);

if (p.needReference())
{
    fvc::makeRelative(phiHbyA, U);
    adjustPhi(phiHbyA, U, p);
    fvc::makeAbsolute(phiHbyA, U);
}

tmp<volScalarField> rAtU(rAU);

if (pimple.consistent())
{
    rAtU = 1.0/max(1.0/rAU - UEqn.H1(), 0.1/rAU);
    phiHbyA +=
        fvc::interpolate(rAtU() - rAU)*fvc::snGrad(p)*mesh.magSf();
    HbyA -= (rAU - rAtU())*fvc::grad(p);
}

if (pimple.nCorrPiso() <= 1)
{
    tUEqn.clear();
}


// Update the pressure BCs to ensure flux consistency
constrainPressure(p, U, phiHbyA, rAtU(), MRF);

// Non-orthogonal pressure corrector loop
while (pimple.correctNonOrthogonal())
{
    fvScalarMatrix pEqn
    (
        fvm::laplacian(rAtU(), p) == fvc::div(phiHbyA)
    );

    pEqn.setReference(pRefCell, pRefValue);

    if (smDiagnosticsActiveThisStep && !smOperatorDiagnosticsDoneThisStep)
    {
        const scalar diagnosticVolume = gSum(mesh.V());
        tmp<scalarField> smResidual = pEqn.residual();
        smOperatorRmsSMThisStep = Foam::sqrt
        (
            gSum(mesh.V()*sqr(smResidual()))/diagnosticVolume
        );

        volScalarField pExtrapCandidate
        (
            IOobject
            (
                "pExtrapCandidate",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE,
                false
            ),
            pExtrap
        );
        surfaceScalarField phiHbyAExtrap
        (
            IOobject
            (
                "phiHbyAExtrap",
                runTime.timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE,
                false
            ),
            phiHbyA
        );
        constrainPressure
        (
            pExtrapCandidate,
            U,
            phiHbyAExtrap,
            rAtU(),
            MRF
        );

        fvScalarMatrix extrapEqn
        (
            fvm::laplacian(rAtU(), pExtrapCandidate)
         == fvc::div(phiHbyAExtrap)
        );
        extrapEqn.setReference(pRefCell, pRefValue);
        tmp<scalarField> extrapResidual = extrapEqn.residual();
        smOperatorRmsExtrapThisStep = Foam::sqrt
        (
            gSum(mesh.V()*sqr(extrapResidual()))/diagnosticVolume
        );
        smOperatorRMSImprovementThisStep = computeImprovementPercent
        (
            smOperatorRmsExtrapThisStep,
            smOperatorRmsSMThisStep
        );
        smOperatorDiagnosticsDoneThisStep = true;
    }

    const SolverPerformance<scalar> pSolvePerformance = pEqn.solve();
    if (smDiagnosticsActiveThisStep && !smFirstGamgResidualCapturedThisStep)
    {
        smFirstGamgInitialResidualThisStep = pSolvePerformance.initialResidual();
        smFirstGamgResidualCapturedThisStep = true;
    }

    if (pimple.finalNonOrthogonalIter())
    {
        phi = phiHbyA - pEqn.flux();
    }
}

#include "continuityErrs.H"

// Explicitly relax pressure for momentum corrector
p.relax();

U = HbyA - rAtU()*fvc::grad(p);
U.correctBoundaryConditions();
fvOptions.correct(U);
