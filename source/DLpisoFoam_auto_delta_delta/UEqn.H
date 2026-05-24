// Solve the Momentum equation

MRF.correctBoundaryVelocity(U);

U_prev = U;

fvVectorMatrix UEqn
(
    fvm::ddt(U) + fvm::div(phi, U)
  + MRF.DDt(U)
  + turbulence->divDevSigma(U)
 ==
    fvOptions(U)
);

UEqn.relax();

fvOptions.constrain(UEqn);

if (piso.momentumPredictor())
{
    solve(UEqn == -fvc::grad(p));

    fvOptions.correct(U);
}

// Storing the velocity increment and divergence fields (only once per time step)
delta_U = U - U_prev;
delta_delta_U = delta_U - delta_U_prev;

// Compute divergence fields for ML surrogate model
div_U = fvc::div(U);
div_dU = fvc::div(delta_U);
div_delta_delta_U = fvc::div(delta_delta_U);
