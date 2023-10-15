
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

/*volVectorField U_non_cons*/
/*(*/
/*    IOobject*/
/*    (*/
/*        "U_non_cons",*/
/*        runTime.timeName(),*/
/*        mesh,*/
/*        IOobject::NO_READ,*/
/*        IOobject::AUTO_WRITE*/
/*    ),*/
/*    mesh,*/
/*    dimensionedVector("U_non_cons", dimensionSet(0,1,-1,0,0,0,0), vector::zero)*/
/*);*/

volScalarField out   // this field is to visualize the output from the surrogate model
(
    IOobject
    (
        "out",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedScalar("out", dimensionSet(0,2,-2,0,0,0,0), 0.0)
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


