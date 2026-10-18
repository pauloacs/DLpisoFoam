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

constrainPressure(p, U, phiHbyA, rAtU(), MRF);

while (pimple.correctNonOrthogonal())
{
    fvScalarMatrix pEqn
    (
        fvm::laplacian(rAtU(), p) == fvc::div(phiHbyA)
    );

    pEqn.setReference(pRefCell, pRefValue);

    const auto pEqnPerf = pEqn.solve();

    // Per-pass GAMG bookkeeping (accumulators declared/reset in the .C file,
    // once per pass, before this include is used) — this is how the twin
    // solver measures whether the injected guess actually reduced work.
    oracleTwinPassTotalPIterations += pEqnPerf.nIterations();
    if (!oracleTwinPassFirstResidualCaptured)
    {
        oracleTwinPassFirstResidual = pEqnPerf.initialResidual();
        oracleTwinPassFirstIterations = pEqnPerf.nIterations();
        oracleTwinPassFirstResidualCaptured = true;
    }

    if (pimple.finalNonOrthogonalIter())
    {
        phi = phiHbyA - pEqn.flux();
    }
}

#include "continuityErrs.H"

p.relax();

U = HbyA - rAtU()*fvc::grad(p);
U.correctBoundaryConditions();
fvOptions.correct(U);
