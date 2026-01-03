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

// volScalarField f_U
// (
//     IOobject
//     (
//         "f_U",
//         runTime.timeName(),
//         mesh,
//         IOobject::NO_READ,
//         IOobject::AUTO_WRITE
//     ),
//     mesh,
//     dimensionedScalar("f_U", dimensionSet(0,0,-2,0,0,0,0), 0.0)
// );

// volScalarField dP_dx
// (
//     IOobject
//     (
//         "dP_dx",
//         runTime.timeName(),
//         mesh,
//         IOobject::NO_READ,
//         IOobject::AUTO_WRITE
//     ),
//     mesh,
//     dimensionedScalar("f_U", dimensionSet(0,1,-2,0,0,0,0), 0.0)
// );

// volScalarField dP_dy
// (
//     IOobject
//     (
//         "dP_dy",
//         runTime.timeName(),
//         mesh,
//         IOobject::NO_READ,
//         IOobject::AUTO_WRITE
//     ),
//     mesh,
//     dimensionedScalar("f_U", dimensionSet(0,1,-2,0,0,0,0), 0.0)
// );


volVectorField U_non_cons
(
    IOobject
    (
        "U_non_cons",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedVector("U_non_cons", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);

volVectorField U_prev
(
    IOobject
    (
        "U_prev",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedVector("U_prev", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);

volVectorField delta_U_prev
(
    IOobject
    (
        "delta_U_prev",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedVector("delta_U_prev", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);

volScalarField p_prev
(
    IOobject
    (
        "p_prev",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("p_prev", dimensionSet(0,2,-2,0,0,0,0), 0.0)
);

volScalarField p_prev_prev
(
    IOobject
    (
        "p_prev",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("p_prev", dimensionSet(0,2,-2,0,0,0,0), 0.0)
);

volScalarField delta_p_prev
(
    IOobject
    (
        "delta_p_prev",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("delta_p_prev", dimensionSet(0,2,-2,0,0,0,0), 0.0)
);

volVectorField delta_U
(
    IOobject
    (
        "delta_U",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedVector("delta_U", dimensionSet(0,1,-1,0,0,0,0), vector::zero)
);

volScalarField delta_p
(
    IOobject
    (
        "delta_p",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("delta_p", dimensionSet(0,2,-2,0,0,0,0), 0.0)
);

#include "createPhi.H"


label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell(p, piso.dict(), pRefCell, pRefValue);
mesh.setFluxRequired(p.name());


singlePhaseTransportModel laminarTransport(U, phi);

autoPtr<incompressible::turbulenceModel> turbulence
(
    incompressible::turbulenceModel::New(U, phi, laminarTransport)
);

#include "createMRF.H"
#include "createFvOptions.H"
