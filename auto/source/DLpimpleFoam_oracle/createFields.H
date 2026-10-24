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
setRefCell(p, mesh.solutionDict().subDict("PIMPLE"), pRefCell, pRefValue);
mesh.setFluxRequired(p.name());

singlePhaseTransportModel laminarTransport(U, phi);

autoPtr<incompressible::momentumTransportModel> turbulence
(
    incompressible::momentumTransportModel::New(U, phi, laminarTransport)
);

#include "createMRF.H"
#include "createFvOptions.H"

// This diagnostic solver assumes a k-omega/k-epsilon style RAS model, whose
// state fields (k, omega, nut) are registered in the mesh objectRegistry by
// the time the turbulence model above is constructed. They are looked up
// (not created) so the twin-pass checkpoint/restore below can rewind them.
volScalarField& oracleTwinK =
    const_cast<volScalarField&>(mesh.lookupObject<volScalarField>("k"));
volScalarField& oracleTwinOmega =
    const_cast<volScalarField&>(mesh.lookupObject<volScalarField>("omega"));
volScalarField& oracleTwinNut =
    const_cast<volScalarField&>(mesh.lookupObject<volScalarField>("nut"));

// ---- History fields (updated exactly once per real timestep) ----

volScalarField p_prevTime
(
    IOobject
    (
        "p_prevTime",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    p
);

volScalarField dpPrevTime
(
    IOobject
    (
        "dpPrevTime",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("dpPrevTime", dimensionSet(0,2,-2,0,0,0,0), 0.0)
);

volScalarField ddpPrevTime
(
    IOobject
    (
        "ddpPrevTime",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("ddpPrevTime", dimensionSet(0,2,-2,0,0,0,0), 0.0)
);

// Diagnostic/inspection copies of the reference (truth) pressure deltas.
volScalarField dpTruth
(
    IOobject
    (
        "dpTruth",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("dpTruth", dimensionSet(0,2,-2,0,0,0,0), 0.0)
);

volScalarField ddpTruth
(
    IOobject
    (
        "ddpTruth",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("ddpTruth", dimensionSet(0,2,-2,0,0,0,0), 0.0)
);

volScalarField pOracleGuess
(
    IOobject
    (
        "pOracleGuess",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    p
);

// Actual dp supplied to the oracle after optional interpolation and weighting.
volScalarField oraclePressureIncrement
(
    IOobject
    (
        "oraclePressureIncrement",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar
    (
        "oraclePressureIncrement",
        p.dimensions(),
        0.0
    )
);

// Perfect dU oracle after optional interpolation and weighting. The field is
// always written for diagnostics and is injected only when requested.
volVectorField oracleVelocityIncrement
(
    IOobject
    (
        "oracleVelocityIncrement",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedVector
    (
        "oracleVelocityIncrement",
        U.dimensions(),
        vector::zero
    )
);
