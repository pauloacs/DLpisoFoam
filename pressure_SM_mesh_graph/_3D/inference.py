import json
from pathlib import Path

import numpy as np

from .graph_io import (feature_names, features_from_fields, fields_from_raw, load_mesh_graph,
                       validate_config, previous_velocity_increment, compose_velocity_increment)
from .graph_model import reconstruct_velocity_shift
from .mesh_graph import coarse_vector_gradient, prolong, restrict


_state = {}


def configured_data_dir(config):
    return Path(config.get('data_folder', config.get('ML_data_folder', 'ML_data')))


def configure(config):
    validate_config(config)
    _state.clear()
    _state['config'] = dict(config)


def init_func(array, z_top_boundary, z_bot_boundary, y_top_boundary, y_bot_boundary,
              obst_boundary, reference_faces=None, mesh_bounds=None):
    if _state['config'].get('predict_dU_output', False):
        import sys
        if getattr(sys.modules.get('python_module'), 'solver_total_dU_contract', 0) != 1:
            raise RuntimeError('Joint dU prediction requires the rebuilt DLpimpleFoam_auto solver.')
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() != 1:
        raise NotImplementedError('Graph prototype currently requires a serial OpenFOAM case.')
    if reference_faces is None or mesh_bounds is None or array.shape[1] != 12:
        raise ValueError('Graph initialization requires the updated eight-argument solver interface.')
    config = _state['config']
    data_dir = configured_data_dir(config)
    metadata = json.loads((data_dir / 'graph_model.json').read_text())
    graph, signature = load_mesh_graph(array[:, 3:6], array[:, 11], Path.cwd(), data_dir,
                                       int(config.get('graph_levels', 3)))
    if (metadata['signature'] != signature or metadata['feature_names'] != feature_names(config)
            or metadata.get('var_to_predict', 'ddp') != config.get('var_to_predict', 'ddp')
            or metadata.get('velocity_var_to_predict', 'dU') != config.get('velocity_var_to_predict', 'dU')
            or metadata.get('predict_dU_output', False) != config.get('predict_dU_output', False)
            or metadata.get('velocity_output_mode', 'direct') != config.get('velocity_output_mode', 'direct')):
        raise ValueError('Graph model mesh/configuration mismatch; retrain with this exact mesh and feature set.')
    from .graph_model import MeshGraphShifter
    from .pressure_gauge import prepare_pressure_gauge
    model = MeshGraphShifter(graph, metadata['channels'], metadata['width'], metadata['steps'],
                             predict_velocity=config.get('predict_dU_output', False),
                             velocity_output_mode=config.get('velocity_output_mode', 'direct'))
    model.load_weights(str(data_dir / 'graph_weights.h5'))
    with np.load(data_dir / 'graph_stats.npz') as stats:
        mean, std, target_scale = stats['mean'], stats['std'], float(stats['target_scale'])
        velocity_scale = float(stats['velocity_scale']) if config.get('predict_dU_output', False) else 1.
    gauge = prepare_pressure_gauge(graph['centres'], reference_faces)
    _state.update(graph=graph, model=model, mean=mean, std=std, target_scale=target_scale, gauge=gauge,
                  velocity_scale=velocity_scale,
                  velocity_shift_length_scale=float(metadata.get('velocity_shift_length_scale', 1.)))
    print(f"[MESH GRAPH] initialized; cells={len(array)} coarse_nodes={graph['coarse_count']}", flush=True)
    return 0


def py_func(array, U_max_norm, simulation_time=None):
    from .graph_model import reconstruct
    from .pressure_gauge import apply_pressure_gauge
    graph, config = _state['graph'], _state['config']
    features, gradient = features_from_fields(fields_from_raw(array, config), U_max_norm, graph, config)
    output = _state['model']((features - _state['mean']) / _state['std'], training=False)
    prediction = reconstruct(output, gradient, _state['target_scale']).numpy().astype(np.float64)
    prediction -= np.average(prediction, weights=graph['coarse_volumes'])
    prediction *= float(U_max_norm) ** 2 * float(config.get('output_weight_factor', 1.))
    if config.get('var_to_predict', 'ddp') == 'ddp' and config.get('ddp_full_coarse_prediction', False):
        prediction += float(config.get('extrapolation_weight_factor', 1.)) * restrict(
            array[:, 10], graph['mapping'], graph['volumes'], graph['coarse_count'])
    result = prolong(prediction, graph, bool(config.get('graph_linear_prolongation', True)))
    apply_pressure_gauge(result, _state['gauge'])
    if not np.isfinite(result).all():
        raise ValueError('Graph inference produced nonfinite CFD pressure.')
    if config.get('inspect_results', False):
        from .plots import plot_pressure_comparison, pressure_components
        count = _state.get('prediction_count', 0) + 1
        _state['prediction_count'] = count
        interval = max(1, int(config.get('inspect_interval', 1)))
        if count == 1 or count % interval == 0:
            predicted_dp, _ = pressure_components(result, array[:, 10], config)
            stamp = f'{float(simulation_time):.9g}' if simulation_time is not None else str(count)
            mode = config.get('var_to_predict', 'ddp')
            plot_pressure_comparison(graph['centres'], graph['volumes'], predicted_dp, array[:, 10],
                                     config.get('inspect_output_dir', 'SM_inspect'), f'prediction_{mode}_{stamp}',
                                     previous_p=array[:, 14])
    if config.get('predict_dU_output', False):
        previous_velocity = previous_velocity_increment(array)
        if config.get('velocity_output_mode', 'direct') == 'shift_source':
            normalization = max(abs(float(U_max_norm) * _state['velocity_scale']), 1e-12)
            previous_coarse = restrict(previous_velocity / normalization, graph['mapping'], graph['volumes'],
                                       graph['coarse_count'])
            velocity_gradient = coarse_vector_gradient(previous_coarse, graph)
            velocity = reconstruct_velocity_shift(output, velocity_gradient,
                                                  _state['velocity_shift_length_scale']).numpy()
            velocity = velocity.astype(np.float64) * _state['velocity_scale'] * float(U_max_norm)
        else:
            velocity = np.asarray(output[:, 4:7]) * _state['velocity_scale'] * float(U_max_norm)
        velocity *= float(config.get('velocity_output_weight_factor', config.get('output_weight_factor', 1.)))
        velocity = np.column_stack([prolong(velocity[:, component], graph,
            bool(config.get('velocity_linear_prolongation', False))) for component in range(3)])
        if config.get('velocity_var_to_predict') == 'ddUCorr':
            velocity = compose_velocity_increment(velocity, array[:, 16:19], config, array[:, 11:14])
        else:
            velocity = compose_velocity_increment(velocity, previous_velocity, config)
        if not np.isfinite(velocity).all():
            raise ValueError('Graph inference produced nonfinite velocity increments.')
        return np.ascontiguousarray(np.column_stack((result, velocity)), dtype=np.float64)
    return np.ascontiguousarray(result[:, None], dtype=np.float64)


def reload_weights():
    config = _state['config']
    _state['model'].load_weights(str(configured_data_dir(config) / 'graph_weights.h5'))