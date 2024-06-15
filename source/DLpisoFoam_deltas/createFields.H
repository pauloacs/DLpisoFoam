
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

volScalarField delta_p_CFD
(
    IOobject
    (
        "delta_p_CFD",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("delta_p_CFD", dimensionSet(0,2,-2,0,0,0,0), 0.0)
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


#include "createMRF.H"
#include "createFvOptions.H"


/*  Extracting Cell center coordinates */

volVectorField C
(
   IOobject
   (
       "C",
       runTime.timeName(),
       mesh,
       IOobject::NO_READ,
       IOobject::NO_WRITE,
       false
    ),
    mesh.C(),
    calculatedFvPatchScalarField::typeName
);


