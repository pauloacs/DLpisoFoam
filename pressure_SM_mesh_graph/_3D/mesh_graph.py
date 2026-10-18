import hashlib

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def restrict(values, mapping, volumes, coarse_count):
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] != len(mapping):
        raise ValueError('Field size does not match mesh cell count.')
    total = np.bincount(mapping, weights=volumes, minlength=coarse_count)
    flat = values.reshape(len(mapping), -1)
    result = np.column_stack([
        np.bincount(mapping, weights=volumes * flat[:, component], minlength=coarse_count) / total
        for component in range(flat.shape[1])
    ])
    return result.reshape((coarse_count, *values.shape[1:]))


def _pair_aggregates(indptr, indices):
    mapping = np.full(len(indptr) - 1, -1, dtype=np.int64)
    aggregate = 0
    for cell in np.argsort(np.diff(indptr), kind='stable'):
        if mapping[cell] >= 0:
            continue
        mapping[cell] = aggregate
        for neighbour in indices[indptr[cell]:indptr[cell + 1]]:
            if mapping[neighbour] < 0:
                mapping[neighbour] = aggregate
                break
        aggregate += 1
    return mapping


def coarsen_graph(centres, volumes, owner, neighbour, levels=3):
    centres = np.asarray(centres, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)
    owner = np.asarray(owner, dtype=np.int64)
    neighbour = np.asarray(neighbour, dtype=np.int64)
    count = len(centres)
    if centres.shape != (count, 3) or volumes.shape != (count,) or not np.all(np.isfinite(centres)):
        raise ValueError('Mesh centres/volumes have invalid shape or values.')
    if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0):
        raise ValueError('Cell volumes must be finite and positive.')
    if owner.shape != neighbour.shape or np.any(owner < 0) or np.any(neighbour < 0):
        raise ValueError('Internal-face owner/neighbour arrays must have identical shapes and valid indices.')
    if np.any(owner >= count) or np.any(neighbour >= count) or levels < 0:
        raise ValueError('Invalid graph connectivity or coarsening level.')
    rows = np.concatenate((owner, neighbour))
    columns = np.concatenate((neighbour, owner))
    keep = rows != columns
    adjacency = sparse.csr_matrix((np.ones(np.count_nonzero(keep)), (rows[keep], columns[keep])), shape=(count, count))
    original_components, _ = connected_components(adjacency)
    mapping = np.arange(count)
    for _ in range(levels):
        local = _pair_aggregates(adjacency.indptr, adjacency.indices)
        mapping = local[mapping]
        coarse_count = int(local.max()) + 1
        edges = adjacency.tocoo()
        keep = local[edges.row] != local[edges.col]
        adjacency = sparse.csr_matrix((np.ones(np.count_nonzero(keep)), (local[edges.row[keep]], local[edges.col[keep]])),
                                      shape=(coarse_count, coarse_count))
        adjacency.data[:] = 1.
    coarse_count = adjacency.shape[0]
    if connected_components(adjacency)[0] != original_components:
        raise ValueError('Coarsening changed the number of disconnected flow regions.')
    coarse_centres = restrict(centres, mapping, volumes, coarse_count)
    coarse_volumes = np.bincount(mapping, weights=volumes, minlength=coarse_count)
    edges = sparse.triu(adjacency, k=1).tocoo()
    source, target = edges.row.astype(np.int64), edges.col.astype(np.int64)
    directed_source = np.concatenate((source, target))
    directed_target = np.concatenate((target, source))
    offsets = coarse_centres[directed_target] - coarse_centres[directed_source]
    distance = np.linalg.norm(offsets, axis=1)
    weights = 1. / np.maximum(distance, 1e-12)
    weights /= np.bincount(directed_source, weights=weights, minlength=coarse_count)[directed_source]
    return dict(mapping=mapping, volumes=volumes, centres=centres, coarse_centres=coarse_centres,
                coarse_volumes=coarse_volumes, source=directed_source, target=directed_target,
                edge_offsets=offsets, edge_weights=weights, coarse_count=coarse_count)


def prolong(values, graph, linear=True):
    values = np.asarray(values, dtype=np.float64)
    mapping = graph['mapping']
    result = values[mapping].copy()
    if not linear:
        return result
    if values.ndim != 1:
        raise ValueError('Linear reconstruction currently supports scalar pressure only.')
    source, target = graph['source'], graph['target']
    offsets = graph['edge_offsets']
    weight = 1. / np.maximum(np.sum(offsets ** 2, axis=1), 1e-24)
    normal = np.zeros((graph['coarse_count'], 3, 3))
    rhs = np.zeros((graph['coarse_count'], 3))
    np.add.at(normal, source, offsets[:, :, None] * offsets[:, None, :] * weight[:, None, None])
    np.add.at(rhs, source, offsets * (weight * (values[target] - values[source]))[:, None])
    gradient = np.einsum('nij,nj->ni', np.linalg.pinv(normal), rhs)
    correction = np.einsum('ni,ni->n', graph['centres'] - graph['coarse_centres'][mapping], gradient[mapping])
    result += correction
    return result


def mesh_signature(centres, volumes, owner, neighbour, levels):
    digest = hashlib.sha256(str(levels).encode('ascii'))
    for array in (centres, volumes, owner, neighbour):
        array = np.ascontiguousarray(array)
        digest.update(str(array.shape).encode('ascii'))
        digest.update(array.tobytes())
    return digest.hexdigest()