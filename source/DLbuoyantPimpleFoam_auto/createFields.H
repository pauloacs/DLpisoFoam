#include "createRDeltaT.H"

Info<< "Reading thermophysical properties\n" << endl;

autoPtr<fluidThermo> pThermo
(
    fluidThermo::New(mesh)
);
fluidThermo& thermo = pThermo();
thermo.validate(args.executable(), "h", "e");

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

volScalarField& p = thermo.p();

volScalarField rho
(
    IOobject
    (
        "rho",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    thermo.rho()
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

#include "compressibleCreatePhi.H"

Info<< "Creating turbulence model\n" << endl;
autoPtr<compressible::momentumTransportModel> turbulence
(
    compressible::momentumTransportModel::New
    (
        rho,
        U,
        phi,
        thermo
    )
);

Info<< "Creating thermophysical transport model\n" << endl;
autoPtr<fluidThermophysicalTransportModel> thermophysicalTransport
(
    fluidThermophysicalTransportModel::New(turbulence(), thermo)
);


#include "readGravitationalAcceleration.H"
#include "readhRef.H"
#include "gh.H"
#include "readpRef.H"


Info<< "Reading field p_rgh\n" << endl;
volScalarField p_rgh
(
    IOobject
    (
        "p_rgh",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

// Force p_rgh to be consistent with p
p_rgh = p - rho*gh - pRef;

pressureControl pressureControl
(
    p,
    p_rgh,
    rho,
    pimple.dict(),
    thermo.incompressible()
);

mesh.setFluxRequired(p_rgh.name());

Info<< "Creating field dpdt\n" << endl;
volScalarField dpdt
(
    IOobject
    (
        "dpdt",
        runTime.timeName(),
        mesh
    ),
    mesh,
    dimensionedScalar(p.dimensions()/dimTime, 0)
);

// Create fields necessary for the Surrogate model


volScalarField delta_p_rgh
(
    IOobject
    (
        "delta_p_rgh",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("delta_p_rgh", dimensionSet(1,-1,-2,0,0,0,0), 0.0)
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

volScalarField p_rgh_prev
(
    IOobject
    (
        "p_rgh_prev",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("p_rgh_prev", dimensionSet(1,-1,-2,0,0,0,0), 0.0)
);


volScalarField delta_p_rgh_CFD
(
    IOobject
    (
        "delta_p_rgh_CFD",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("delta_p_rgh_CFD", dimensionSet(1,-1,-2,0,0,0,0), 0.0)
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


Info<< "Creating field kinetic energy K\n" << endl;
volScalarField K("K", 0.5*magSqr(U));

dimensionedScalar initialMass = fvc::domainIntegrate(rho);

#include "createMRF.H"
#include "createRadiationModel.H"
#include "createFvOptions.H"
