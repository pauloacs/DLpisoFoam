// Solve the Momentum equation

MRF.correctBoundaryVelocity(U);

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

if (smRefreshInjectedPressurePatches)
{
    const volScalarField patchRAU(1.0/UEqn.A());
    const volVectorField patchHbyA(constrainHbyA(patchRAU*UEqn.H(), U, p));
    surfaceScalarField patchPhi
    (
        "smDirectPatchPhi",
        fvc::flux(patchHbyA) + MRF.zeroFilter(fvc::interpolate(patchRAU)*fvc::ddtCorr(U, phi))
    );
    MRF.makeRelative(patchPhi);
    if (p.needReference())
    {
        fvc::makeRelative(patchPhi, U);
        adjustPhi(patchPhi, U, p);
        fvc::makeAbsolute(patchPhi, U);
    }
    constrainPressure(p, U, patchPhi, patchRAU, MRF);
    p.correctBoundaryConditions();
    pSM = p;
    smRefreshInjectedPressurePatches = false;
    Info<< "[SM DIRECT] injected pressure patches constrained before fresh momentum solve." << nl;
}

if (pimple.momentumPredictor())
{
    solve(UEqn == -fvc::grad(p));

    fvOptions.correct(U);
}

