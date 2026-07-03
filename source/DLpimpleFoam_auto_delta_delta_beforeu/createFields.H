#include "createRDeltaT.H"

Info<< "Reading field p\n" << endl;
volScalarField p
(
    IOobject
    (
        "p",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

#include "createPhi.H"


label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell(p, pimple.dict(), pRefCell, pRefValue);
mesh.setFluxRequired(p.name());


singlePhaseTransportModel laminarTransport(U, phi);

autoPtr<incompressible::momentumTransportModel> turbulence
(
    incompressible::momentumTransportModel::New(U, phi, laminarTransport)
);

#include "createMRF.H"
#include "createFvOptions.H"

// ---- Fields for the Surrogate Model ----

volScalarField p_prev
(
    IOobject
    (
        "p_prev",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("p_prev", dimensionSet(0,2,-2,0,0,0,0), 0.0)
);

volScalarField dpML
(
    IOobject
    (
        "dpML",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("dpML", dimensionSet(0,2,-2,0,0,0,0), 0.0)
);

volScalarField ddpML
(
    IOobject
    (
        "ddpML",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("ddpML", dimensionSet(0,2,-2,0,0,0,0), 0.0)
);

volScalarField ddpPrev
(
    IOobject
    (
        "ddpPrev",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("ddpPrev", dimensionSet(0,2,-2,0,0,0,0), 0.0)
);

volScalarField ddp
(
    IOobject
    (
        "ddp",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("ddp", dimensionSet(0,2,-2,0,0,0,0), 0.0)
);

volScalarField dpPrev
(
    IOobject
    (
        "dpPrev",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("dpPrev", dimensionSet(0,2,-2,0,0,0,0), 0.0)
);

volScalarField dp
(
    IOobject
    (
        "dp",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("dp", dimensionSet(0,2,-2,0,0,0,0), 0.0)
);

volVectorField grad_dpPrev
(
    IOobject
    (
        "grad_dpPrev",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("grad_dpPrev", dimensionSet(0,1,-2,0,0,0,0), vector::zero)
);

volScalarField laplace_dpPrev
(
    IOobject
    (
        "laplace_dpPrev",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("laplace_dpPrev", dimensionSet(0,0,-2,0,0,0,0), 0.0)
);

volScalarField uDotGradDpPrev
(
    IOobject
    (
        "uDotGradDpPrev",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("uDotGradDpPrev", dimensionSet(0,2,-3,0,0,0,0), 0.0)
);

volScalarField gradDpPrevMag
(
    IOobject
    (
        "gradDpPrevMag",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("gradDpPrevMag", dimensionSet(0,1,-2,0,0,0,0), 0.0)
);

// PISO pressure-equation quantities exported as ML inputs.
// Computed right after the first momentum predictor (UEqn) call.
// rAU_ML    = 1/A(U)          [s]
// HbyA_ML   = rAU_ML * H(U)   [m/s]  (constrained)
// divHbyA_ML = div(HbyA_ML)   [1/s]
volScalarField rAU_ML
(
    IOobject
    (
        "rAU_ML",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("rAU_ML", dimensionSet(0,0,1,0,0,0,0), 0.0)
);

volVectorField HbyA_ML
(
    IOobject
    (
        "HbyA_ML",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("HbyA_ML", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);

volScalarField divHbyA_ML
(
    IOobject
    (
        "divHbyA_ML",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("divHbyA_ML", dimensionSet(0,0,-1,0,0,0,0), 0.0)
);

// Previous time-step values of HbyA_ML / divHbyA_ML (internal storage used to
// build the temporal variations dHbyA_ML / dDivHbyA_ML).
volVectorField HbyA_ML_prev
(
    IOobject
    (
        "HbyA_ML_prev",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("HbyA_ML_prev", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);

volScalarField divHbyA_ML_prev
(
    IOobject
    (
        "divHbyA_ML_prev",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("divHbyA_ML_prev", dimensionSet(0,0,-1,0,0,0,0), 0.0)
);

// Temporal variations (current - previous time step).
// dHbyA_ML    = HbyA_ML - HbyA_ML_prev       [m/s]
// dDivHbyA_ML = divHbyA_ML - divHbyA_ML_prev [1/s]
volVectorField dHbyA_ML
(
    IOobject
    (
        "dHbyA_ML",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("dHbyA_ML", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);

volScalarField dDivHbyA_ML
(
    IOobject
    (
        "dDivHbyA_ML",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("dDivHbyA_ML", dimensionSet(0,0,-1,0,0,0,0), 0.0)
);

// Additional pressure-equation inputs
// rAUGradDpPrev_ML      = rAU_ML * grad(dpPrev)                    [m/s]
// divRAUGradDpPrev_ML   = div(rAU_ML * grad(dpPrev))               [1/s]
// pressureEqResidualp_ML = div(HbyA_ML) - div(rAU_ML*grad(dpPrev)) [1/s]
// rAUGradpPrev_ML       = rAU_ML * grad(p_prev)                    [m/s]
// divRAUGradpPrev_ML    = div(rAU_ML * grad(p_prev))               [1/s]
volVectorField rAUGradDpPrev_ML
(
    IOobject
    (
        "rAUGradDpPrev_ML",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("rAUGradDpPrev_ML", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);

volScalarField divRAUGradDpPrev_ML
(
    IOobject
    (
        "divRAUGradDpPrev_ML",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("divRAUGradDpPrev_ML", dimensionSet(0,0,-1,0,0,0,0), 0.0)
);

volScalarField pressureEqResidualp_ML
(
    IOobject
    (
        "pressureEqResidualp_ML",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("pressureEqResidualp_ML", dimensionSet(0,0,-1,0,0,0,0), 0.0)
);

volVectorField rAUGradpPrev_ML
(
    IOobject
    (
        "rAUGradpPrev_ML",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("rAUGradpPrev_ML", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);

volScalarField divRAUGradpPrev_ML
(
    IOobject
    (
        "divRAUGradpPrev_ML",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("divRAUGradpPrev_ML", dimensionSet(0,0,-1,0,0,0,0), 0.0)
);

volVectorField grad_dp
(
    IOobject
    (
        "grad_dp",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("grad_dp", dimensionSet(0,1,-2,0,0,0,0), vector::zero)
);

volVectorField UStarPrev
(
    IOobject
    (
        "UStarPrev",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("UStarPrev", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);

volVectorField UStar
(
    IOobject
    (
        "UStar",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("UStar", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);

volVectorField dUStar
(
    IOobject
    (
        "dUStar",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("dUStar", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);

volVectorField dUStarPrev
(
    IOobject
    (
        "dUStarPrev",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("dUStarPrev", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);

volVectorField ddUStar
(
    IOobject
    (
        "ddUStar",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("ddUStar", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);

volVectorField ddUStarPrev
(
    IOobject
    (
        "ddUStarPrev",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("ddUStarPrev", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);

volScalarField divUFirstPred
(
    IOobject
    (
        "divUFirstPred",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("divUFirstPred", dimensionSet(0,0,-1,0,0,0,0), 0.0)
);

// Pressure-correction velocity increment: dUCorr = U - UStar  (velocity gained during pressure solve)
volVectorField dUCorr
(
    IOobject
    (
        "dUCorr",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("dUCorr", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);

// Second difference of pressure-correction increment — training target: ddUCorr = dUCorr - dUCorrPrev
volVectorField ddUCorr
(
    IOobject
    (
        "ddUCorr",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("ddUCorr", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);

// U at the start of the current time step (before any UEqn call) — used to compute dUCorr via UStar
volVectorField U_time_prev
(
    IOobject
    (
        "U_time_prev",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("U_time_prev", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);

// Previous time-step dUCorr — used to compute ddUCorr
volVectorField dUCorrPrev
(
    IOobject
    (
        "dUCorrPrev",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("dUCorrPrev", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);

// Previous time-step ddUCorr — available as extra feature if needed
volVectorField ddUCorrPrev
(
    IOobject
    (
        "ddUCorrPrev",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("ddUCorrPrev", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);

// ML-predicted second non-conservative velocity increment — applied as starting-guess correction in pEqn
// Zero until the surrogate becomes active and predict_ddU is enabled in python_module.
volVectorField ddUStarML
(
    IOobject
    (
        "ddUStarML",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("ddUStarML", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);
