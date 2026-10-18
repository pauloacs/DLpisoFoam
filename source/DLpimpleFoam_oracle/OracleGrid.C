#include "OracleGrid.H"

tmp<volScalarField> oracleGridRoundTrip
(
    const volScalarField& fieldIn,
    const vector& gridRes
)
{
    const fvMesh& mesh = fieldIn.mesh();

    // Global (parallel-reduced) bounding box of the mesh points.
    const boundBox bb(mesh.points(), true);
    const vector span(bb.max() - bb.min());

    // Node counts (>=2 per direction) spanning the domain at gridRes spacing.
    const label nx = max(label(2), label(ceil(span.x()/gridRes.x())) + 1);
    const label ny = max(label(2), label(ceil(span.y()/gridRes.y())) + 1);
    const label nz = max(label(2), label(ceil(span.z()/gridRes.z())) + 1);
    const label nNodes = nx*ny*nz;

    // For a given point, returns the lower-corner node index (i0,j0,k0) and
    // the trilinear fractional weights (tx,ty,tz) within that node's cell.
    auto lowerCornerAndFrac = [&](const point& p, label& i0, label& j0, label& k0,
                                  scalar& tx, scalar& ty, scalar& tz)
    {
        const scalar fx = (p.x() - bb.min().x())/gridRes.x();
        const scalar fy = (p.y() - bb.min().y())/gridRes.y();
        const scalar fz = (p.z() - bb.min().z())/gridRes.z();

        i0 = min(max(label(floor(fx)), 0), nx - 2);
        j0 = min(max(label(floor(fy)), 0), ny - 2);
        k0 = min(max(label(floor(fz)), 0), nz - 2);

        tx = min(max(fx - i0, 0.0), 1.0);
        ty = min(max(fy - j0, 0.0), 1.0);
        tz = min(max(fz - k0, 0.0), 1.0);
    };

    auto nodeIndex = [&](label i, label j, label k) -> label
    {
        return i + nx*(j + ny*k);
    };

    const vectorField& C = mesh.C();
    const scalarField& V = mesh.V();

    // Restriction: trilinear-weighted scatter of cell values onto the 8
    // surrounding nodes (a smooth analogue of building the ML background
    // grid from CFD cell data).
    scalarField nodeSum(nNodes, 0.0);
    scalarField nodeWeight(nNodes, 0.0);

    forAll(C, cellI)
    {
        label i0, j0, k0;
        scalar tx, ty, tz;
        lowerCornerAndFrac(C[cellI], i0, j0, k0, tx, ty, tz);

        for (label di = 0; di < 2; ++di)
        {
            const scalar wx = di ? tx : (1 - tx);
            for (label dj = 0; dj < 2; ++dj)
            {
                const scalar wxy = wx*(dj ? ty : (1 - ty));
                for (label dk = 0; dk < 2; ++dk)
                {
                    const scalar w = wxy*(dk ? tz : (1 - tz));
                    const label idx = nodeIndex(i0 + di, j0 + dj, k0 + dk);
                    nodeSum[idx] += w*V[cellI]*fieldIn[cellI];
                    nodeWeight[idx] += w*V[cellI];
                }
            }
        }
    }

    // Combine per-processor partial sums so every rank ends up with the
    // same global node values before reconstructing local cell values.
    if (Pstream::parRun())
    {
        Pstream::listCombineGather(nodeSum, plusEqOp<scalar>());
        Pstream::listCombineGather(nodeWeight, plusEqOp<scalar>());
        Pstream::listCombineScatter(nodeSum);
        Pstream::listCombineScatter(nodeWeight);
    }

    scalarField nodeValue(nNodes, 0.0);
    forAll(nodeValue, idx)
    {
        if (nodeWeight[idx] > SMALL)
        {
            nodeValue[idx] = nodeSum[idx]/nodeWeight[idx];
        }
    }

    tmp<volScalarField> tFieldOut
    (
        new volScalarField
        (
            IOobject
            (
                fieldIn.name() + "OracleGrid",
                mesh.time().timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            fieldIn
        )
    );
    volScalarField& fieldOut = tFieldOut.ref();

    // Prolongation: smooth 8-node trilinear reconstruction at each cell
    // (fully local — node values are already broadcast to every rank).
    forAll(C, cellI)
    {
        label i0, j0, k0;
        scalar tx, ty, tz;
        lowerCornerAndFrac(C[cellI], i0, j0, k0, tx, ty, tz);

        scalar value = 0.0;
        for (label di = 0; di < 2; ++di)
        {
            const scalar wx = di ? tx : (1 - tx);
            for (label dj = 0; dj < 2; ++dj)
            {
                const scalar wxy = wx*(dj ? ty : (1 - ty));
                for (label dk = 0; dk < 2; ++dk)
                {
                    const scalar w = wxy*(dk ? tz : (1 - tz));
                    value += w*nodeValue[nodeIndex(i0 + di, j0 + dj, k0 + dk)];
                }
            }
        }
        fieldOut[cellI] = value;
    }
    fieldOut.correctBoundaryConditions();

    return tFieldOut;
}


tmp<surfaceScalarField> oracleGridRoundTrip
(
    const surfaceScalarField& fieldIn,
    const vector& gridRes
)
{
    const fvMesh& mesh = fieldIn.mesh();
    const boundBox bb(mesh.points(), true);
    const vector span(bb.max() - bb.min());

    const label nx = max(label(2), label(ceil(span.x()/gridRes.x())) + 1);
    const label ny = max(label(2), label(ceil(span.y()/gridRes.y())) + 1);
    const label nz = max(label(2), label(ceil(span.z()/gridRes.z())) + 1);
    const label nNodes = nx*ny*nz;

    auto lowerCornerAndFrac = [&](const point& p, label& i0, label& j0, label& k0,
                                  scalar& tx, scalar& ty, scalar& tz)
    {
        const scalar fx = (p.x() - bb.min().x())/gridRes.x();
        const scalar fy = (p.y() - bb.min().y())/gridRes.y();
        const scalar fz = (p.z() - bb.min().z())/gridRes.z();

        i0 = min(max(label(floor(fx)), 0), nx - 2);
        j0 = min(max(label(floor(fy)), 0), ny - 2);
        k0 = min(max(label(floor(fz)), 0), nz - 2);

        tx = min(max(fx - i0, 0.0), 1.0);
        ty = min(max(fy - j0, 0.0), 1.0);
        tz = min(max(fz - k0, 0.0), 1.0);
    };

    auto nodeIndex = [&](label i, label j, label k) -> label
    {
        return i + nx*(j + ny*k);
    };

    const vectorField& Cf = mesh.Cf();
    const scalarField magSf(mag(mesh.Sf()));
    scalarField nodeSum(nNodes, 0.0);
    scalarField nodeWeight(nNodes, 0.0);

    // Restrict internal-face values to Cartesian nodes using trilinear and
    // face-area weighting. Boundary values are reconstructed separately
    // below from the same global node field.
    forAll(fieldIn, facei)
    {
        label i0, j0, k0;
        scalar tx, ty, tz;
        lowerCornerAndFrac(Cf[facei], i0, j0, k0, tx, ty, tz);

        for (label di = 0; di < 2; ++di)
        {
            const scalar wx = di ? tx : (1 - tx);
            for (label dj = 0; dj < 2; ++dj)
            {
                const scalar wxy = wx*(dj ? ty : (1 - ty));
                for (label dk = 0; dk < 2; ++dk)
                {
                    const scalar w = wxy*(dk ? tz : (1 - tz));
                    const label idx = nodeIndex(i0 + di, j0 + dj, k0 + dk);
                    nodeSum[idx] += w*magSf[facei]*fieldIn[facei];
                    nodeWeight[idx] += w*magSf[facei];
                }
            }
        }
    }

    // Include physical boundary faces in the restriction.
    forAll(fieldIn.boundaryField(), patchi)
    {
        const fvsPatchScalarField& pf = fieldIn.boundaryField()[patchi];
        const vectorField& pCf = mesh.Cf().boundaryField()[patchi];
        const scalarField pMagSf(mag(mesh.Sf().boundaryField()[patchi]));

        forAll(pf, patchFacei)
        {
            label i0, j0, k0;
            scalar tx, ty, tz;
            lowerCornerAndFrac
            (
                pCf[patchFacei], i0, j0, k0, tx, ty, tz
            );

            for (label di = 0; di < 2; ++di)
            {
                const scalar wx = di ? tx : (1 - tx);
                for (label dj = 0; dj < 2; ++dj)
                {
                    const scalar wxy = wx*(dj ? ty : (1 - ty));
                    for (label dk = 0; dk < 2; ++dk)
                    {
                        const scalar w = wxy*(dk ? tz : (1 - tz));
                        const label idx =
                            nodeIndex(i0 + di, j0 + dj, k0 + dk);
                        nodeSum[idx] +=
                            w*pMagSf[patchFacei]*pf[patchFacei];
                        nodeWeight[idx] += w*pMagSf[patchFacei];
                    }
                }
            }
        }
    }

    if (Pstream::parRun())
    {
        Pstream::listCombineGather(nodeSum, plusEqOp<scalar>());
        Pstream::listCombineGather(nodeWeight, plusEqOp<scalar>());
        Pstream::listCombineScatter(nodeSum);
        Pstream::listCombineScatter(nodeWeight);
    }

    scalarField nodeValue(nNodes, 0.0);
    forAll(nodeValue, idx)
    {
        if (nodeWeight[idx] > SMALL)
        {
            nodeValue[idx] = nodeSum[idx]/nodeWeight[idx];
        }
    }

    tmp<surfaceScalarField> tFieldOut
    (
        new surfaceScalarField
        (
            IOobject
            (
                fieldIn.name() + "OracleGrid",
                mesh.time().timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE,
                false
            ),
            fieldIn
        )
    );
    surfaceScalarField& fieldOut = tFieldOut.ref();

    auto interpolateAtPoint = [&](const point& p) -> scalar
    {
        label i0, j0, k0;
        scalar tx, ty, tz;
        lowerCornerAndFrac(p, i0, j0, k0, tx, ty, tz);

        scalar value = 0.0;
        for (label di = 0; di < 2; ++di)
        {
            const scalar wx = di ? tx : (1 - tx);
            for (label dj = 0; dj < 2; ++dj)
            {
                const scalar wxy = wx*(dj ? ty : (1 - ty));
                for (label dk = 0; dk < 2; ++dk)
                {
                    const scalar w = wxy*(dk ? tz : (1 - tz));
                    value +=
                        w*nodeValue[nodeIndex(i0 + di, j0 + dj, k0 + dk)];
                }
            }
        }
        return value;
    };

    forAll(fieldOut, facei)
    {
        fieldOut[facei] = interpolateAtPoint(Cf[facei]);
    }

    forAll(fieldOut.boundaryFieldRef(), patchi)
    {
        fvsPatchScalarField& pf = fieldOut.boundaryFieldRef()[patchi];
        const vectorField& pCf = mesh.Cf().boundaryField()[patchi];
        forAll(pf, patchFacei)
        {
            pf[patchFacei] = interpolateAtPoint(pCf[patchFacei]);
        }
    }

    return tFieldOut;
}
