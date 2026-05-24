// Solve the Momentum equation

MRF.correctBoundaryVelocity(U);

U_prev = U;

tmp<fvVectorMatrix> tUEqn
(
    fvm::ddt(U) + fvm::div(phi, U)
  + MRF.DDt(U)
  + turbulence->divDevSigma(U)
 ==
    fvOptions(U)
);
fvVectorMatrix& UEqn = tUEqn.ref();

UEqn.relax();

fvOptions.constrain(UEqn);

if (pimple.momentumPredictor())
{
    solve(UEqn == -fvc::grad(p));

    fvOptions.correct(U);
}

// Storing the velocity increment and divergence fields
// ONLY at the first PIMPLE iteration
// (These must be consistent inputs for the ML model)
if (pimple.firstPimpleIter())
{
    delta_U = U - U_prev;
    delta_delta_U = delta_U - delta_U_prev;

    // Compute divergence fields for ML surrogate model
    div_U = fvc::div(U);
    div_dU = fvc::div(delta_U);
    div_delta_delta_U = fvc::div(delta_delta_U);
}
