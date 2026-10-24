#ifndef OracleGrid_H
#define OracleGrid_H

#include "fvCFD.H"

// Self-contained uniform-grid trilinear round trip: a proxy for the
// resolution loss a coarse ML grid would introduce even on a perfect
// prediction. Cell values are volume+trilinear-weighted scattered onto a
// uniform Cartesian node grid at gridRes spacing (parallel-reduced across
// all processors, i.e. a restriction), then reconstructed at each mesh cell
// via smooth 8-node trilinear interpolation (a prolongation) — matching the
// production ML pipeline's 'trilinear' NPtoOF reconstruction method, not a
// blocky/piecewise-constant approximation.
tmp<volScalarField> oracleGridRoundTrip
(
    const volScalarField& fieldIn,
    const vector& gridRes
);

// Equivalent uniform-grid round trip for face-based scalar fields such as
// phi or dPhi. Restriction uses face-area and trilinear weighting.
tmp<surfaceScalarField> oracleGridRoundTrip
(
    const surfaceScalarField& fieldIn,
    const vector& gridRes
);

#endif
