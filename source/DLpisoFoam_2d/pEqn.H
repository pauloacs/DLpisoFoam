volScalarField rAU(1.0/UEqn.A());
volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));
surfaceScalarField phiHbyA
(
    "phiHbyA",
    fvc::flux(HbyA)
  + MRF.zeroFilter(fvc::interpolate(rAU)*fvc::ddtCorr(U, phi))
);

MRF.makeRelative(phiHbyA);

adjustPhi(phiHbyA, U, p);

if (counter == 0) {
    // Estimating p(t) based on p(t-1) + delta_p (as estimated by the SM)
    p = p  + delta_p; //(delta_p*0.5 + delta_p_CFD*0.5);
}

// Update the pressure BCs to ensure flux consistency
constrainPressure(p, U, phiHbyA, rAU, MRF);

// Non-orthogonal pressure corrector loop
while (piso.correctNonOrthogonal())
{
    // Pressure corrector

    fvScalarMatrix pEqn
    (
        fvm::laplacian(rAU, p) == fvc::div(phiHbyA)
    );

    pEqn.setReference(pRefCell, pRefValue);

    pEqn.solve();

    if (piso.finalNonOrthogonalIter())
    {
        phi = phiHbyA - pEqn.flux();
    }
}

    // Pressure corrector


#include "continuityErrs.H"

U = HbyA - rAU*fvc::grad(p);
U.correctBoundaryConditions();
fvOptions.correct(U);