import numpy as np
from scipy.spatial import cKDTree


def prepare_pressure_gauge(centres, reference_faces):
    centres = np.asarray(centres, dtype=np.float64)
    faces = np.asarray(reference_faces, dtype=np.float64)
    if centres.ndim != 2 or centres.shape[1] != 3 or len(centres) < 4 or not np.isfinite(centres).all():
        raise ValueError('Pressure gauge requires at least four finite mesh cell centres.')
    if faces.ndim != 2 or faces.shape[1] != 7 or not len(faces) or not np.isfinite(faces).all():
        raise ValueError('Pressure gauge requires finite reference faces [centre(3), normal(3), area].')
    if np.any(faces[:, 6] <= 0) or np.any(np.linalg.norm(faces[:, 3:6], axis=1) == 0):
        raise ValueError('Reference faces require positive areas and nonzero normals.')
    tree = cKDTree(centres)
    stencils = []
    target = np.array([1., 0., 0., 0.])
    limit = min(len(centres), 4096)
    for face in faces:
        count = min(64, limit)
        while True:
            distances, candidates = tree.query(face[:3], k=count)
            offsets = centres[candidates] - face[:3]
            scale = max(float(distances[-1]), np.finfo(float).eps)
            inward = offsets @ (face[3:6] / np.linalg.norm(face[3:6])) <= 1e-10 * scale
            candidates, offsets = candidates[inward], offsets[inward] / scale
            design = np.column_stack((np.ones(len(candidates)), offsets))
            proximity = 1. / np.maximum(np.linalg.norm(offsets, axis=1), 1e-3)
            coefficients = np.linalg.lstsq((design * proximity[:, None]).T, target, rcond=None)[0]
            weights = coefficients * proximity
            if np.allclose(weights @ design, target, rtol=0., atol=1e-8):
                stencils.append((candidates, weights))
                break
            if count == limit:
                raise ValueError('Mesh geometry cannot support affine pressure-reference extrapolation.')
            count = min(2 * count, limit)
    count = max(len(candidates) for candidates, _ in stencils)
    vertices = np.zeros((len(faces), count), dtype=np.int64)
    weights = np.zeros((len(faces), count))
    for index, (candidates, face_weights) in enumerate(stencils):
        vertices[index, :len(candidates)] = candidates
        weights[index, :len(candidates)] = face_weights
    return vertices, weights, faces[:, 6].copy()


def apply_pressure_gauge(field, gauge):
    vertices, weights, areas = gauge
    correction = np.average(np.sum(field.reshape(-1)[vertices] * weights, axis=1), weights=areas)
    field -= correction
    return correction