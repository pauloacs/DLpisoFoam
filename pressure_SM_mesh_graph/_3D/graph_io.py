from pathlib import Path

import numpy as np
from foamlib import FoamFile

from .mesh_graph import coarsen_graph, mesh_signature, restrict


# Legacy feature names follow the 49-column solver interface: `U` is UStar,
# and `ddUStar` is the predictor increment difference, not the total-ddU target.
# `ddUStarDiff` is ddUStar - ddUStarPrev (a further predictor difference).
FIELD_LAYOUT = (
    ('add_U_input', 'U', 0, 3, 1),
    ('add_dUStar_input', 'dUStar', 11, 14, 1),
    ('add_ddUStar_input', 'ddUStar', 3, 6, 1),
    ('add_ddUStarDiff_input', 'ddUStarDiff', 3, 6, 1),
    ('add_ddUPrev_input', 'ddUPrev', None, None, 1),
    ('add_divUStar_input', 'divUFirstPred', 15, 16, 1),
    ('add_dUCorrPrev_input', 'dUCorrPrev', 16, 19, 1),
    ('add_ddUCorrPrev_input', 'ddUCorrPrev', 19, 22, 1),
    ('add_ddpPrev_input', 'ddpPrev', 9, 10, 2),
    ('add_dpPrev_input', 'dpPrev', 10, 11, 2),
    ('add_pPrev_input', 'p_prev', 14, 15, 2),
    ('add_gradDpPrev_input', 'gradDpPrev', 22, 25, 2),
    ('add_laplacian_dpPrev_input', 'laplaceDpPrev', 25, 26, 2),
    ('add_uDotGradDpPrev_input', 'uDotGradDpPrev', 26, 27, 3),
    ('add_gradDpPrevMag_input', 'gradDpPrevMag', 27, 28, 2),
    ('include_rAU_input', 'rAU', 28, 29, 0),
    ('include_HbyA_input', 'HbyA', 29, 32, 1),
    ('include_divHbyA_input', 'divHbyA', 32, 33, 1),
    ('add_pressureEqResidualp_input', 'pressureEqResidualp', 41, 42, 1),
    ('include_dHbyA_input', 'dHbyA', 33, 36, 1),
    ('include_dDivHbyA_input', 'dDivHbyA', 36, 37, 1),
    ('add_rAUGradDpPrev_input', 'rAUGradDpPrev', 37, 40, 1),
    ('add_divRAUGradDpPrev_input', 'divRAUGradDpPrev', 40, 41, 1),
    ('add_rAUGradpPrev_input', 'rAUGradpPrev', 42, 45, 1),
    ('add_divRAUGradpPrev_input', 'divRAUGradpPrev', 45, 46, 1),
)


def coarse_volume_weights(coarse_volumes, power=1.0):
    volumes = np.asarray(coarse_volumes, dtype=np.float64)
    power = float(power)
    if (volumes.ndim != 1 or not len(volumes) or not np.isfinite(volumes).all()
            or np.any(volumes <= 0)):
        raise ValueError('coarse_volumes must be a nonempty vector of finite positive values.')
    if not np.isfinite(power) or not 0 <= power <= 1:
        raise ValueError('graph_volume_weight_power must be finite and in [0,1].')
    relative_volumes = volumes / np.mean(volumes)
    weights = np.power(relative_volumes, power)
    return weights / weights.sum()


def validate_config(config):
    volume_power = float(config.get('graph_volume_weight_power', 1.0))
    if not np.isfinite(volume_power) or not 0 <= volume_power <= 1:
        raise ValueError('graph_volume_weight_power must be finite and in [0,1].')
    graph_loss = config.get('graph_loss', 'coarse_mse')
    if graph_loss not in ('coarse_mse', 'native_obstacle_gradient', 'coarse_native_blend'):
        raise ValueError('graph_loss must be coarse_mse, native_obstacle_gradient, or coarse_native_blend.')
    if graph_loss in ('native_obstacle_gradient', 'coarse_native_blend'):
        fraction = float(config.get('loss_obstacle_fraction', .5))
        radius = float(config.get('loss_obstacle_radius', .05))
        if not np.isfinite(fraction) or not 0 <= fraction <= 1 or not np.isfinite(radius) or radius <= 0:
            raise ValueError('Invalid obstacle loss fraction or radius.')
        if graph_loss == 'coarse_native_blend':
            native_fraction = float(config.get('graph_native_loss_fraction', .25))
            if not np.isfinite(native_fraction) or not 0 <= native_fraction <= 1:
                raise ValueError('graph_native_loss_fraction must be finite and in [0,1].')
        for name, default in (('loss_pressure_gradient_weight', .2), ('loss_velocity_gradient_weight', .1)):
            if not np.isfinite(float(config.get(name, default))) or float(config.get(name, default)) < 0:
                raise ValueError(f'{name} must be finite and nonnegative.')
    if config.get('var_to_predict', 'ddp') not in ('dp', 'ddp'):
        raise ValueError('var_to_predict must be dp or ddp.')
    if config.get('predict_ddUCorr_output', False) or config.get('oracle_mode', False):
        raise ValueError('Use predict_dU_output for total velocity increments; legacy ddUCorr/oracle overrides are unsupported.')
    velocity_output_mode = config.get('velocity_output_mode', 'direct')
    if velocity_output_mode not in ('direct', 'shift_source'):
        raise ValueError('velocity_output_mode must be direct or shift_source.')
    if velocity_output_mode == 'shift_source' and not config.get('predict_dU_output', False):
        raise ValueError('shift_source velocity output requires predict_dU_output=True.')
    if config.get('use_feature_decomposition', False):
        raise ValueError('The graph prototype uses mesh nodes, not Tucker cores.')
    if not config.get('add_gradDpPrev_input', False):
        raise ValueError('The graph shifter requires add_gradDpPrev_input=True.')
    for name, default in (('graph_width', 24), ('graph_steps', 4), ('graph_patience', 20), ('num_epochs', 200)):
        value = config.get(name, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    rate = float(config.get('lr', 5e-4))
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError('lr must be finite and positive.')
    for name in ('output_weight_factor', 'extrapolation_weight_factor'):
        if not np.isfinite(float(config.get(name, 1.))):
            raise ValueError(f'{name} must be finite.')
    if config.get('predict_dU_output', False):
        if config.get('velocity_var_to_predict', 'dU') not in ('dU', 'ddU', 'ddUCorr'):
            raise ValueError('velocity_var_to_predict must be dU, ddU, or ddUCorr.')
        if config.get('velocity_var_to_predict') == 'ddUCorr':
            if graph_loss != 'coarse_mse':
                raise ValueError('ddUCorr currently requires coarse_mse.')
            if not config.get('add_dUStar_input') or not config.get('add_dUCorrPrev_input'):
                raise ValueError('ddUCorr requires add_dUStar_input and add_dUCorrPrev_input.')
        if velocity_output_mode == 'shift_source':
            if config.get('velocity_var_to_predict', 'dU') != 'ddU':
                raise ValueError('shift_source velocity output currently requires velocity_var_to_predict=ddU.')
            if not config.get('add_dUPrev_input', False):
                raise ValueError('shift_source velocity output requires add_dUPrev_input=True.')
            if graph_loss != 'coarse_mse':
                raise ValueError('shift_source velocity output currently supports graph_loss=coarse_mse only.')
        weight = float(config.get('velocity_loss_weight', 1.))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError('velocity_loss_weight must be finite and positive.')
        if not np.isfinite(float(config.get('velocity_output_weight_factor', config.get('output_weight_factor', 1.)))):
            raise ValueError('velocity_output_weight_factor must be finite.')
        if not np.isfinite(float(config.get('velocity_extrapolation_weight_factor', 1.))):
            raise ValueError('velocity_extrapolation_weight_factor must be finite.')
    supported = {entry[0] for entry in FIELD_LAYOUT}
    supported.add('add_dUPrev_input')
    for name, value in config.items():
        if name.endswith('_input') and name.startswith(('add_', 'include_')) and value and name not in supported:
            raise ValueError(f'Input {name} is not implemented by the graph prototype.')


def load_mesh_graph(centres, volumes, case_dir, data_dir, levels=3):
    mesh = Path(case_dir) / 'constant/polyMesh'
    all_owner = np.asarray(FoamFile(mesh / 'owner')[None], dtype=np.int64)
    neighbour = np.asarray(FoamFile(mesh / 'neighbour')[None], dtype=np.int64)
    owner = all_owner[:len(neighbour)]
    boundary = FoamFile(mesh / 'boundary')[None]
    signature = mesh_signature(centres, volumes, all_owner, neighbour, (levels, boundary))
    cache = Path(data_dir) / 'mesh_graph.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as stored:
            if str(stored['signature']) == signature:
                graph = {name: stored[name] for name in stored.files if name != 'signature'}
                graph['coarse_count'] = int(graph['coarse_count'])
                graph['native_owner'] = owner
                graph['native_neighbour'] = neighbour
                print(f"[MESH GRAPH] loaded {len(centres)} -> {graph['coarse_count']} nodes", flush=True)
                return graph, signature
    graph = coarsen_graph(centres, volumes, owner, neighbour, levels)
    boundary_features = np.zeros((len(centres), 3))
    for name, patch in boundary:
        first, count = int(patch['startFace']), int(patch['nFaces'])
        if patch.get('type') == 'wall':
            boundary_features[all_owner[first:first + count], 0] = 1.
        elif name == 'inlet':
            boundary_features[all_owner[first:first + count], 1] = 1.
        elif name == 'outlet':
            boundary_features[all_owner[first:first + count], 2] = 1.
    graph['boundary_features'] = restrict(boundary_features, graph['mapping'], volumes, graph['coarse_count'])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, signature=signature, **graph)
    print(f"[MESH GRAPH] built {len(centres)} -> {graph['coarse_count']} nodes; "
          f"directed edges={len(graph['source'])}", flush=True)
    return graph, signature


def field_layout(config):
    layout = tuple((flag, 'gradpPrev', 46, 49, exponent)
                 if field == 'gradDpPrev' and config.get('var_to_predict', 'ddp') == 'dp'
                 else (flag, field, first, last, exponent)
                 for flag, field, first, last, exponent in FIELD_LAYOUT)
    if config.get('add_dUPrev_input', False):
        layout += (('add_dUPrev_input', 'dUPrev', None, None, 1),)
    return layout


def feature_names(config):
    return [field for flag, field, *_ in field_layout(config) if config.get(flag, False)]


def features_from_fields(fields, scale, graph, config):
    scale = max(float(scale), 1e-12)
    parts = []
    gradient = None
    for flag, field, _, _, exponent in field_layout(config):
        if not config.get(flag, False):
            continue
        values = restrict(np.asarray(fields[field]) / scale ** exponent,
                          graph['mapping'], graph['volumes'], graph['coarse_count'])
        values = values.reshape(graph['coarse_count'], -1)
        if field in ('ddpPrev', 'dpPrev', 'p_prev'):
            values -= np.average(values, axis=0, weights=graph['coarse_volumes'])
        if field in ('gradDpPrev', 'gradpPrev'):
            gradient = values.copy()
        parts.append(values)
    centres = graph['coarse_centres']
    span = np.maximum(np.ptp(graph['centres'], axis=0), 1e-12)
    parts.append((centres - graph['centres'].min(axis=0)) / span)
    parts.append(np.log(graph['coarse_volumes'] / np.mean(graph['coarse_volumes']))[:, None])
    parts.append(graph['boundary_features'])
    features = np.concatenate(parts, axis=1)
    if not np.isfinite(features).all() or gradient is None:
        raise ValueError('Invalid graph features or missing pressure gradient.')
    return features.astype(np.float32), gradient.astype(np.float32)


def fields_from_raw(raw, config):
    if raw.ndim != 2 or raw.shape[1] != 49:
        raise ValueError('Expected the existing CFD interface with 49 raw columns.')
    fields = {field: raw[:, first:last] for flag, field, first, last, _ in field_layout(config)
              if config.get(flag, False) and field not in ('dUPrev', 'ddUPrev', 'ddUStarDiff')}
    if config.get('add_ddUStarDiff_input', False):
        fields['ddUStarDiff'] = raw[:, 3:6] - raw[:, 6:9]
    if config.get('add_dUPrev_input', False):
        fields['dUPrev'] = previous_velocity_increment(raw)
    if config.get('add_ddUPrev_input', False):
        fields['ddUPrev'] = previous_velocity_second_increment(raw)
    return fields


def previous_velocity_increment(raw):
    return raw[:, 11:14] - raw[:, 3:6] + raw[:, 16:19]


def previous_velocity_second_increment(raw):
    """Return previous total ddU = ddUStarPrev + ddUCorrPrev."""
    return raw[:, 6:9] + raw[:, 19:22]


def velocity_target_from_fields(fields, config):
    """Return dU = Uf - U0 directly, or ddU = dU - dUPrev."""
    mode = config.get('velocity_var_to_predict', 'dU')
    if mode in fields:
        return np.asarray(fields[mode])
    if mode == 'ddU' and 'dU' in fields and 'dUPrev' in fields:
        return np.asarray(fields['dU']) - np.asarray(fields['dUPrev'])
    if mode == 'ddUCorr' and all(name in fields for name in ('dU', 'dUStar', 'dUCorrPrev')):
        return np.asarray(fields['dU']) - fields['dUStar'] - fields['dUCorrPrev']
    raise ValueError(f'Missing {mode} velocity target; collect fresh velocity-history samples.')


def compose_velocity_increment(weighted_prediction, previous, config, predictor_increment=None):
    """Convert a predicted ddU residual to total dU once; direct dU passes through."""
    if config.get('velocity_var_to_predict', 'dU') == 'ddU':
        return weighted_prediction + float(config.get('velocity_extrapolation_weight_factor', 1.)) * previous
    if config.get('velocity_var_to_predict') == 'ddUCorr':
        if predictor_increment is None:
            raise ValueError('ddUCorr composition requires the current dUStar.')
        return (weighted_prediction + predictor_increment
                + float(config.get('velocity_extrapolation_weight_factor', 1.)) * previous)
    return weighted_prediction


def target_from_fields(fields, config):
    mode = config.get('var_to_predict', 'ddp')
    if mode in fields:
        return np.asarray(fields[mode]).reshape(-1)
    if mode == 'dp' and 'ddp' in fields and 'dpPrev' in fields:
        return np.asarray(fields['ddp']).reshape(-1) + np.asarray(fields['dpPrev']).reshape(-1)
    raise ValueError(f'Missing {mode} training target.')