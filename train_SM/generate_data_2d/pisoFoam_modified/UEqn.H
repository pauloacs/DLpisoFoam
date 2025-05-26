// Solve the Momentum equation

MRF.correctBoundaryVelocity(U);

U_prev = U;

fvVectorMatrix UEqn
(
    fvm::ddt(U) + fvm::div(phi, U)
  + MRF.DDt(U)
  + turbulence->divDevReff(U)
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

//Before running pEqn.H this writes U_non_cons and f_U = f(U_non_cons)

U_non_cons = U;

delta_U = U_non_cons - U_prev;

// volTensorField gradU = fvc::grad(U);

// volScalarField du_dx = gradU.component(tensor::XX);
// volScalarField dv_dx = gradU.component(tensor::XY);
// volScalarField du_dy = gradU.component(tensor::YX);
// volScalarField dv_dy = gradU.component(tensor::YY);

// f_U = pow(du_dx, 2.0) + pow(dv_dy, 2.0) + 2.0 * du_dy * dv_dx;

