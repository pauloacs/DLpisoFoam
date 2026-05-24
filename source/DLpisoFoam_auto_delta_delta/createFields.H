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
setRefCell(p, piso.dict(), pRefCell, pRefValue);
mesh.setFluxRequired(p.name());

singlePhaseTransportModel laminarTransport(U, phi);

autoPtr<incompressible::momentumTransportModel> turbulence
(
    incompressible::momentumTransportModel::New(U, phi, laminarTransport)
);

// Cell center coordinates for surrogate model
volVectorField C
(
    IOobject
    (
        "C",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh.C()
);

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

volScalarField delta_p
(
    IOobject
    (
        "delta_p",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("delta_p", dimensionSet(0,2,-2,0,0,0,0), 0.0)
);

volScalarField delta_delta_p
(
    IOobject
    (
        "delta_delta_p",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("delta_delta_p", dimensionSet(0,2,-2,0,0,0,0), 0.0)
);

volScalarField delta_delta_p_prev
(
    IOobject
    (
        "delta_delta_p_prev",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("delta_delta_p_prev", dimensionSet(0,2,-2,0,0,0,0), 0.0)
);

volScalarField delta_delta_p_CFD
(
    IOobject
    (
        "delta_delta_p_CFD",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("delta_delta_p_CFD", dimensionSet(0,2,-2,0,0,0,0), 0.0)
);

volScalarField delta_p_prev
(
    IOobject
    (
        "delta_p_prev",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("delta_p_prev", dimensionSet(0,2,-2,0,0,0,0), 0.0)
);

volScalarField delta_p_CFD
(
    IOobject
    (
        "delta_p_CFD",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("delta_p_CFD", dimensionSet(0,2,-2,0,0,0,0), 0.0)
);

volVectorField U_prev
(
    IOobject
    (
        "U_prev",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("U_prev", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);

volVectorField delta_U
(
    IOobject
    (
        "delta_U",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("delta_U", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);

volVectorField delta_U_prev
(
    IOobject
    (
        "delta_U_prev",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("delta_U_prev", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);

volVectorField delta_delta_U
(
    IOobject
    (
        "delta_delta_U",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("delta_delta_U", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);

volVectorField delta_delta_U_prev
(
    IOobject
    (
        "delta_delta_U_prev",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("delta_delta_U_prev", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);

// Divergence fields for ML surrogate model
volScalarField div_U
(
    IOobject
    (
        "div_U",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("div_U", dimensionSet(0,0,-1,0,0,0,0), 0.0)
);

volScalarField div_dU
(
    IOobject
    (
        "div_dU",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("div_dU", dimensionSet(0,0,-1,0,0,0,0), 0.0)
);

volScalarField div_delta_delta_U
(
    IOobject
    (
        "div_delta_delta_U",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("div_delta_delta_U", dimensionSet(0,0,-1,0,0,0,0), 0.0)
);

#include "createMRF.H"
#include "createFvOptions.H"
