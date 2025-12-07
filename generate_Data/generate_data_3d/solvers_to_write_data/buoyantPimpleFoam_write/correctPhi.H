// CorrectPhi
// (
//     U,
//     phi,
//     p_rgh,
//     rho,
//     psi,
//     dimensionedScalar("rAUf", dimTime, 1),
//     divrhoU(),
//     pimple,
//     true
// );

#ifndef CORRECTPHI_H
#define CORRECTPHI_H

// Include necessary headers
#include "volFields.H"
#include "surfaceFields.H"
#include "dimensionedScalar.H"
#include "pimpleControl.H"

// Declare the CorrectPhi function
void CorrectPhi(
        volVectorField& U,
        surfaceScalarField& phi,
        volScalarField& p_rgh,
        volScalarField& rho,
        volScalarField& psi,
        const dimensionedScalar& rAUf,
        const surfaceScalarField& divrhoU,
        pimpleControl& pimple,
        bool finalIter
);

#endif // CORRECTPHI_H