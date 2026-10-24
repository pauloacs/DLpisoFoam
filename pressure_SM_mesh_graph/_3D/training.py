import argparse
import csv
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path

import h5py
import numpy as np

from .graph_io import (coarse_volume_weights, feature_names, features_from_fields, load_mesh_graph, target_from_fields,
                       validate_config, velocity_target_from_fields, compose_velocity_increment)
from .mesh_graph import coarse_vector_gradient, restrict


def update_retraining_index(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ('update', 'status', 'started_utc', 'completed_utc', 'samples', 'training_samples',
              'validation_samples', 'features', 'pressure_target', 'velocity_target',
              'epochs_requested', 'learning_rate', 'pre_update_validation', 'best_validation',
              'checkpoint_updated', 'output_directory')
    records = []
    if path.exists():
        with path.open(newline='') as handle:
            records = list(csv.DictReader(handle))
    records = [row for row in records if row.get('update') != str(record['update'])]
    records.append(record)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in records)
    temporary.replace(path)


def native_validation_prediction(pressure, velocity, group, graph, config, linear):
    from .mesh_graph import prolong
    previous = np.asarray(group['dpPrev'][:], dtype=float).reshape(-1)
    truth = np.asarray(group['dp'][:], dtype=float).reshape(-1) if config.get('var_to_predict', 'ddp') == 'dp' else (
        np.asarray(group['ddp'][:], dtype=float).reshape(-1) + previous)
    pressure = pressure.copy()
    full = config.get('var_to_predict', 'ddp') == 'ddp' and config.get('ddp_full_coarse_prediction', False)
    if full:
        pressure += float(config.get('extrapolation_weight_factor', 1.)) * restrict(
            previous, graph['mapping'], graph['volumes'], graph['coarse_count'])
    pressure = prolong(pressure, graph, linear=linear)
    if config.get('var_to_predict', 'ddp') == 'ddp' and not full:
        pressure += float(config.get('extrapolation_weight_factor', 1.)) * previous
    for field in (pressure, truth, previous):
        field -= np.average(field, weights=graph['volumes'])
    result = dict(prediction=pressure, truth=truth, previous=previous)
    if velocity is not None:
        velocity = np.column_stack([prolong(velocity[:, component], graph, linear=linear) for component in range(3)])
        previous_velocity = group['dUPrev'][:] if 'dUPrev' in group else np.zeros_like(velocity)
        if config.get('velocity_var_to_predict') == 'ddUCorr':
            velocity = compose_velocity_increment(velocity, group['dUCorrPrev'][:], config, group['dUStar'][:])
        else:
            velocity = compose_velocity_increment(velocity, previous_velocity, config)
        result.update(velocity=velocity, velocity_truth=group['dU'][:], previous_velocity=previous_velocity,
                      start_velocity=group['U'][:] - group['dUStar'][:] if 'U' in group and 'dUStar' in group else None)
    return result


def evaluate_validation(model, samples, sample_keys, graph, stats, config, source, plot_directory):
    from .graph_model import reconstruct, reconstruct_velocity_shift
    from .plots import (comparison_metrics, summarize_comparisons, print_comparison_summary,
                        plot_pressure_comparison, plot_prediction_statistics, plot_shifter_components,
                        plot_validation_history, obstacle_distance_pressure_metrics,
                        plot_obstacle_pressure_error_map, plot_obstacle_distance_pressure_summary,
                        plot_resolution_error_distribution, plot_velocity_comparison)
    plot_directory = Path(plot_directory)
    plot_directory.mkdir(parents=True, exist_ok=True)
    mode = config.get('var_to_predict', 'ddp')
    joint = config.get('predict_dU_output', False)
    records = []
    native_records = []
    obstacle_error_records = []
    with h5py.File(source, 'r') as handle:
        obstacle_points = None
        if 'boundary_coordinates' in handle and 'boundary_patches' in handle:
            patch_name = config.get('loss_obstacle_patch', 'obstacle')
            patch_ids = [int(name.split('_')[-1]) for name, value in handle.attrs.items()
                         if name.startswith('patch_')
                         and (value.decode() if isinstance(value, bytes) else str(value)) == patch_name]
            if patch_ids:
                patch_labels = handle['boundary_patches'][:]
                obstacle_points = handle['boundary_coordinates'][:][np.isin(patch_labels, patch_ids)]
        obstacle_distance = None
        if obstacle_points is not None and len(obstacle_points):
            from scipy.spatial import cKDTree
            obstacle_distance = cKDTree(obstacle_points).query(graph['centres'])[0]
        else:
            print('[MESH GRAPH] obstacle-distance diagnostics skipped: sampled obstacle patch unavailable.', flush=True)
        for index, ((features, gradient, label), name) in enumerate(zip(samples, sample_keys)):
            group = handle[name]
            scale = max(float(np.asarray(group['U_MAX_NORM'][()]).reshape(-1)[0]), 1e-12)
            previous = restrict(group['dpPrev'][:].reshape(-1), graph['mapping'], graph['volumes'], graph['coarse_count'])
            output = model((features - stats['mean']) / stats['std'], training=False)
            predicted = reconstruct(output, gradient, stats['target_scale']).numpy().astype(float)
            predicted -= np.average(predicted, weights=graph['coarse_volumes'])
            pressure_scale = scale ** 2 * float(config.get('output_weight_factor', 1.))
            predicted *= pressure_scale
            pressure_residual = predicted.copy()
            pressure_label = label[:, 0] if joint else label
            truth = pressure_label.astype(float) * scale ** 2
            if mode == 'ddp':
                predicted += float(config.get('extrapolation_weight_factor', 1.)) * previous
                truth += previous
            for field in (predicted, previous, truth):
                field -= np.average(field, weights=graph['coarse_volumes'])
            metrics = comparison_metrics(predicted, previous, truth, graph['coarse_volumes'], mode)
            native_pressure_error = None
            velocity_residual = None
            if joint:
                if config.get('velocity_output_mode', 'direct') == 'shift_source':
                    previous_native = group['dUPrev'][:]
                    previous_coarse = restrict(previous_native / scale, graph['mapping'], graph['volumes'], graph['coarse_count'])
                    velocity_gradient = coarse_vector_gradient(previous_coarse / stats['velocity_scale'], graph)
                    velocity_residual = reconstruct_velocity_shift(
                        output, velocity_gradient, stats['velocity_shift_length_scale']).numpy()
                    velocity_residual *= stats['velocity_scale'] * scale
                else:
                    velocity_residual = output[:, 4:7].numpy() * stats['velocity_scale'] * scale
                velocity_residual *= float(config.get('velocity_output_weight_factor', config.get('output_weight_factor', 1.)))
                velocity_residual_truth = label[:, 1:4] * scale
                velocity = velocity_residual
                velocity_truth = velocity_residual_truth.copy()
                previous_velocity = None
                start_velocity = None
                if 'U' in group and 'dUStar' in group:
                    start_velocity = restrict(group['U'][:] - group['dUStar'][:], graph['mapping'],
                                              graph['volumes'], graph['coarse_count'])
                if 'dUPrev' in group:
                    previous_velocity = restrict(group['dUPrev'][:], graph['mapping'], graph['volumes'], graph['coarse_count'])
                    if config.get('velocity_var_to_predict') == 'ddUCorr':
                        predictor_increment = restrict(group['dUStar'][:], graph['mapping'], graph['volumes'], graph['coarse_count'])
                        correction_previous = restrict(group['dUCorrPrev'][:], graph['mapping'], graph['volumes'], graph['coarse_count'])
                        velocity = compose_velocity_increment(velocity, correction_previous, config, predictor_increment)
                        velocity_truth = restrict(group['dU'][:], graph['mapping'], graph['volumes'], graph['coarse_count'])
                    else:
                        velocity = compose_velocity_increment(velocity, previous_velocity, config)
                    if config.get('velocity_var_to_predict', 'dU') == 'ddU':
                        velocity_truth += previous_velocity
                    metrics['velocity_trend_rmse'] = float(np.sqrt(np.average(np.sum((previous_velocity - velocity_truth) ** 2, axis=1),
                        weights=graph['coarse_volumes'])))
                metrics['velocity_rmse'] = float(np.sqrt(np.average(np.sum((velocity - velocity_truth) ** 2, axis=1),
                    weights=graph['coarse_volumes'])))
                component_errors = velocity - velocity_truth
                for component, axis_name in enumerate('xyz'):
                    metrics[f'velocity_{axis_name}_rmse'] = float(np.sqrt(np.average(
                        component_errors[:, component] ** 2, weights=graph['coarse_volumes'])))
                if config.get('velocity_var_to_predict', 'dU') == 'ddU':
                    residual_errors = velocity_residual - velocity_residual_truth
                    metrics['ddU_rmse'] = float(np.sqrt(np.average(np.sum(residual_errors ** 2, axis=1),
                        weights=graph['coarse_volumes'])))
                metrics['velocity_start_rmse'] = float(np.sqrt(np.average(np.sum(velocity_truth ** 2, axis=1),
                    weights=graph['coarse_volumes'])))
                metrics['u_prev_rmse'] = metrics['velocity_start_rmse']
                if 'velocity_trend_rmse' in metrics:
                    metrics['u_prev_plus_dUPrev_rmse'] = metrics['velocity_trend_rmse']
            if mode in group:
                full_directory = plot_directory / 'full_resolution'
                distribution_fields = [dict(label='Coarse graph', volumes=graph['coarse_volumes'],
                    pressure_error=predicted - truth, velocity_error=velocity - velocity_truth if joint else None)]
                for linear in (False, True):
                    native = native_validation_prediction(pressure_residual, velocity_residual, group, graph, config, linear)
                    native_metrics = comparison_metrics(native['prediction'], native['previous'], native['truth'],
                                                        graph['volumes'], mode)
                    pressure_error = native['prediction'] - native['truth']
                    velocity_error = native['velocity'] - native['velocity_truth'] if joint else None
                    if joint:
                        native_metrics['velocity_rmse'] = float(np.sqrt(np.average(np.sum(velocity_error ** 2, axis=1),
                                                                                  weights=graph['volumes'])))
                    native_records.append(dict(sample=name, linear_prolongation=linear, **native_metrics))
                    distribution_fields.append(dict(label=f'CFD mesh, prolongation={linear}', volumes=graph['volumes'],
                                                     pressure_error=pressure_error, velocity_error=velocity_error))
                    if linear == bool(config.get('graph_linear_prolongation', True)) and obstacle_distance is not None:
                        native_pressure_error = pressure_error
                        obstacle_error_records.extend(obstacle_distance_pressure_metrics(
                            name, pressure_error, native['previous'] - native['truth'], obstacle_distance,
                            graph['centres'], graph['volumes']))
                    if config.get('plot_training', True):
                        directory = full_directory / f'prolongation_{str(linear).lower()}'
                        directory.mkdir(parents=True, exist_ok=True)
                        plot_name = f'validation_{name}'
                        plot_prediction_statistics(native['prediction'], native['previous'], native['truth'],
                            graph['volumes'], directory, plot_name, mesh_label='whole full-resolution CFD mesh')
                        if index < 3:
                            plot_pressure_comparison(graph['centres'], graph['volumes'], native['prediction'], native['previous'],
                                directory, plot_name, truth_dp=native['truth'], mean_free=True, prediction_mode=mode, write_metrics=False)
                        if joint:
                            plot_velocity_comparison(graph['centres'], native['velocity'], native['velocity_truth'], directory,
                                plot_name, previous=native['previous_velocity'], volumes=graph['volumes'],
                                start_velocity=native['start_velocity'], mesh_label='whole full-resolution CFD mesh')
                if config.get('plot_training', True):
                    plot_resolution_error_distribution(distribution_fields, full_directory, f'validation_{name}')
            records.append(dict(sample=name, **metrics))
            if config.get('plot_training', True) and index < 3:
                plot_name = f'validation_{name}'
                plot_pressure_comparison(graph['coarse_centres'], graph['coarse_volumes'], predicted, previous,
                                         plot_directory, plot_name, truth_dp=truth, mean_free=True,
                                         prediction_mode=mode, write_metrics=False)
                for axis, fraction in ((0, .5), (1, .5), (2, .25), (2, .75)):
                    plot_pressure_comparison(graph['coarse_centres'], graph['coarse_volumes'], predicted, previous,
                                             plot_directory, f'{plot_name}_{"xyz"[axis]}_{fraction:g}',
                                             truth_dp=truth, mean_free=True, slice_axis=axis,
                                             slice_fraction=fraction, prediction_mode=mode, write_metrics=False)
                plot_prediction_statistics(predicted, previous, truth, graph['coarse_volumes'], plot_directory, plot_name)
                if native_pressure_error is not None:
                    plot_obstacle_pressure_error_map(graph['centres'], native_pressure_error,
                                                     obstacle_points, plot_directory, plot_name)
                plot_shifter_components(graph['coarse_centres'], output.numpy(), gradient, stats['target_scale'],
                                        pressure_scale, plot_directory, plot_name)
                if joint:
                    plot_velocity_comparison(
                        graph['coarse_centres'], velocity, velocity_truth, plot_directory, plot_name,
                        previous=previous_velocity,
                        residual_prediction=velocity_residual if config.get('velocity_var_to_predict', 'dU') == 'ddU' else None,
                        residual_truth=velocity_residual_truth if config.get('velocity_var_to_predict', 'dU') == 'ddU' else None,
                        volumes=graph['coarse_volumes'], start_velocity=start_velocity
                    )
    if native_records:
        full_directory = plot_directory / 'full_resolution'
        full_directory.mkdir(parents=True, exist_ok=True)
        with (full_directory / 'prediction_metrics.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(native_records[0]))
            writer.writeheader()
            writer.writerows(native_records)
        (full_directory / 'scope.json').write_text(json.dumps(dict(
            pressure_gauge='volume_mean_free', units='physical',
            scope='all native CFD cells after Python transfer and baseline composition, before OpenFOAM boundary correction/reconstruction',
            comparison='same weights and samples; linear prolongation true versus false for both pressure and velocity',
            prolongation='bounded, aggregate-conservative linear reconstruction with three weak gradient-referenced interface steps'), indent=2))
    if obstacle_error_records:
        with (plot_directory / 'obstacle_pressure_error_by_distance.csv').open('w', newline='') as handle:
            fields = ('sample', 'distance_band', 'cells', 'volume_fraction', 'mean_distance_m',
                      'pressure_rmse', 'pressure_mae', 'baseline_rmse', 'improvement_percent')
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(obstacle_error_records)
        plot_obstacle_distance_pressure_summary(obstacle_error_records, plot_directory)
    summary = summarize_comparisons(records, mode)
    summary.update(output_weight_factor=float(config.get('output_weight_factor', 1.)),
                   extrapolation_weight_factor=float(config.get('extrapolation_weight_factor', 1.)))
    if joint:
        summary['velocity_rmse'] = float(np.sqrt(np.mean([record['velocity_rmse'] ** 2 for record in records])))
        summary['velocity_start_rmse'] = float(np.sqrt(np.mean([record['velocity_start_rmse'] ** 2 for record in records])))
        baseline = summary['velocity_start_rmse']
        summary['velocity_improvement_percent'] = 100 * (1 - summary['velocity_rmse'] / baseline) if baseline else None
        message = f"[MESH GRAPH] velocity dU: SM vector RMSE={summary['velocity_rmse']:.6g}, "
        message += f"unchanged-U RMSE={baseline:.6g}, improvement={summary['velocity_improvement_percent']}%"
        print(message, flush=True)
        if 'velocity_trend_rmse' in records[0]:
            trend = float(np.sqrt(np.mean([record['velocity_trend_rmse'] ** 2 for record in records])))
            summary['velocity_trend_rmse'] = trend
            summary['velocity_improvement_vs_trend_percent'] = 100 * (1 - summary['velocity_rmse'] / trend) if trend else None
            message = f"[MESH GRAPH] velocity vs dUPrev trend: baseline RMSE={trend:.6g}, "
            message += f"improvement={summary['velocity_improvement_vs_trend_percent']}%"
            print(message, flush=True)
    with (plot_directory / 'prediction_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    serializable = {key: None if isinstance(value, float) and not np.isfinite(value) else value
                    for key, value in summary.items()}
    (plot_directory / 'baseline_summary.json').write_text(json.dumps(serializable, indent=2, allow_nan=False))
    if config.get('plot_training', True):
        plot_validation_history(records, plot_directory)
    print_comparison_summary(summary)
    print(f'[MESH GRAPH] held-out diagnostics: {plot_directory}', flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train one whole-domain coarse mesh-graph per CFD timestep.')
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--data_file', default=None, help='Read existing samples from this file; write model artifacts only to data_dir.')
    parser.add_argument('--epochs', type=int, default=None, help='Override the case epoch limit for a short training probe.')
    parser.add_argument('--resume', action='store_true', help='Fine-tune existing weights with fixed normalization.')
    parser.add_argument('--evaluate_only', action='store_true', help='Regenerate held-out diagnostics without changing weights, statistics or metadata.')
    parser.add_argument('--plot_dir', default=None, help='Diagnostic output directory for evaluate_only.')
    parser.add_argument('--window_frames', type=int, default=None, help='Use only the latest N timestep samples.')
    args = parser.parse_args(argv)
    if args.evaluate_only and args.resume:
        parser.error('evaluate_only and resume cannot be combined.')
    if args.plot_dir and not args.evaluate_only:
        parser.error('plot_dir requires evaluate_only.')
    config = dict(vars(importlib.import_module('python_module')))
    args.data_dir = args.data_dir or config.get('data_folder', config.get('ML_data_folder', 'ML_data'))
    if args.resume:
        config['num_epochs'] = config.get('retrain_epochs', config.get('num_epochs', 200))
        config['lr'] = config.get('retrain_lr', config.get('lr', 5e-4))
    if args.epochs is not None:
        config['num_epochs'] = args.epochs
    if int(config.get('num_epochs', 200)) < 1:
        parser.error('Epoch count must be positive.')
    if args.window_frames is not None and args.window_frames < 3:
        parser.error('window_frames must be at least three.')
    validate_config(config)
    joint = config.get('predict_dU_output', False)
    graph_loss = config.get('graph_loss', 'coarse_mse')
    native_enabled = graph_loss in ('native_obstacle_gradient', 'coarse_native_blend')
    native_samples = []
    data_dir = Path(args.data_dir)
    source = Path(args.data_file) if args.data_file else data_dir / 'data.h5'
    with h5py.File(source, 'r') as handle:
        required = ('coordinates', 'cell_volumes', 'mesh_bounds')
        if any(name not in handle for name in required):
            raise ValueError('Fresh solver samples with coordinates, volumes and mesh bounds are required.')
        centres, volumes = handle['coordinates'][:], handle['cell_volumes'][:]
        graph, signature = load_mesh_graph(centres, volumes, Path.cwd(), data_dir,
                                           int(config.get('graph_levels', 3)))
        if native_enabled:
            from foamlib import FoamFile
            from .native_loss import native_loss_geometry
            neighbour = np.asarray(FoamFile(Path.cwd() / 'constant/polyMesh/neighbour')[None], dtype=np.int64)
            owner = np.asarray(FoamFile(Path.cwd() / 'constant/polyMesh/owner')[None], dtype=np.int64)[:len(neighbour)]
            obstacle_points = None
            if float(config.get('loss_obstacle_fraction', .5)) > 0:
                patch_name = config.get('loss_obstacle_patch', 'obstacle')
                patch_ids = [int(name.split('_')[-1]) for name, value in handle.attrs.items()
                             if name.startswith('patch_') and (value.decode() if isinstance(value, bytes) else str(value)) == patch_name]
                if not patch_ids or 'boundary_coordinates' not in handle or 'boundary_patches' not in handle:
                    raise ValueError(f'Missing sampled obstacle boundary geometry for {patch_name}.')
                labels = handle['boundary_patches'][:]
                obstacle_points = handle['boundary_coordinates'][:][np.isin(labels, patch_ids)]
            native_geometry = native_loss_geometry(graph, owner, neighbour, config, obstacle_points)
            print('[MESH GRAPH] native loss ROI: ' + json.dumps(native_geometry['diagnostics']), flush=True)
        sample_keys = sorted((name for name in handle if name.startswith('sample_')),
                             key=lambda name: float(handle[name].attrs.get('timestep', name.split('_')[-1])))
        if args.window_frames is not None:
            sample_keys = sample_keys[-args.window_frames:]
        if len(sample_keys) < 3:
            raise ValueError('At least three timestep samples are required for graph training/validation.')
        samples = []
        previous_velocity_coarse = []
        for name in sample_keys:
            group = handle[name]
            if 'dpPrev' not in group:
                raise ValueError('Baseline diagnostics require dpPrev samples; enable add_dpPrev_input.')
            fields = {field: group[field][:] for field in feature_names(config)}
            scale = float(np.asarray(group['U_MAX_NORM'][()]).reshape(-1)[0])
            features, gradient = features_from_fields(fields, scale, graph, config)
            mode = config.get('var_to_predict', 'ddp')
            target_fields = {mode: group[mode][:]} if mode in group else {
                field: group[field][:] for field in ('ddp', 'dpPrev') if field in group}
            label = restrict(target_from_fields(target_fields, config) / max(scale, 1e-12) ** 2,
                             graph['mapping'], volumes, graph['coarse_count'])
            label -= np.average(label, weights=graph['coarse_volumes'])
            if not np.isfinite(label).all():
                raise ValueError(f'Nonfinite graph target in {name}.')
            if joint:
                if 'dU' not in group:
                    raise ValueError('Joint training requires fresh dU samples from the updated solver.')
                if config.get('velocity_output_mode', 'direct') == 'shift_source' and 'dUPrev' not in group:
                    raise ValueError('The shift_source velocity head requires fresh dUPrev samples.')
                velocity_fields = {'dU': group['dU'][:]}
                if config.get('velocity_var_to_predict') == 'ddUCorr':
                    velocity_fields.update(dUStar=group['dUStar'][:], dUCorrPrev=group['dUCorrPrev'][:])
                if 'dUPrev' in group:
                    velocity_fields['dUPrev'] = group['dUPrev'][:]
                    if config.get('velocity_output_mode', 'direct') == 'shift_source':
                        previous_velocity_coarse.append(restrict(
                            group['dUPrev'][:] / max(scale, 1e-12), graph['mapping'], volumes,
                            graph['coarse_count']).astype(np.float32))
                velocity_label = restrict(velocity_target_from_fields(velocity_fields, config) / max(scale, 1e-12),
                                          graph['mapping'], volumes, graph['coarse_count'])
                if velocity_label.shape != (graph['coarse_count'], 3) or not np.isfinite(velocity_label).all():
                    raise ValueError(f'Invalid dU target in {name}.')
                label = np.column_stack((label, velocity_label))
            samples.append((features, gradient, label.astype(np.float32)))
            if native_enabled:
                norm = max(scale, 1e-12)
                previous = group['dpPrev'][:].reshape(-1) / norm ** 2
                pressure = target_from_fields(target_fields, config) / norm ** 2
                if mode == 'ddp':
                    pressure += previous
                native_samples.append(dict(scale=np.float32(norm), pressure=pressure.astype(np.float32), pressure_previous=previous.astype(np.float32),
                    velocity=(velocity_fields['dU'] / norm).astype(np.float32) if joint else np.zeros((len(volumes), 3), np.float32),
                    velocity_previous=(velocity_fields.get('dUPrev', np.zeros((len(volumes), 3))) / norm).astype(np.float32)
                    if joint else np.zeros((len(volumes), 3), np.float32)))
    split = max(1, min(len(samples) - 1, int(.8 * len(samples))))
    training, validation = samples[:split], samples[split:]
    mean = np.mean([features.mean(axis=0) for features, _, _ in training], axis=0)
    variance = np.mean([np.mean((features - mean) ** 2, axis=0) for features, _, _ in training], axis=0)
    std = np.maximum(np.sqrt(variance), 1e-7)
    target_scale = max(float(np.std(np.concatenate([label[:, 0] if joint else label for _, _, label in training]))), 1e-7)
    velocity_scale = max(float(np.sqrt(np.mean(np.concatenate([label[:, 1:4] for _, _, label in training]) ** 2))), 1e-7) if joint else 1.
    velocity_output_mode = config.get('velocity_output_mode', 'direct')
    velocity_shift_length_scale = max(float(np.median(np.linalg.norm(graph['edge_offsets'], axis=1))), 1e-12)
    velocity_gradients = ([coarse_vector_gradient(previous / velocity_scale, graph).astype(np.float32)
                           for previous in previous_velocity_coarse]
                          if joint and velocity_output_mode == 'shift_source' else [])
    metadata = dict(signature=signature, feature_names=feature_names(config),
                    var_to_predict=config.get('var_to_predict', 'ddp'), levels=int(config.get('graph_levels', 3)),
                    width=int(config.get('graph_width', 24)), steps=int(config.get('graph_steps', 4)),
                    channels=len(mean), samples=len(samples), training=split, validation=len(validation),
                    predict_dU_output=bool(joint), velocity_var_to_predict=config.get('velocity_var_to_predict', 'dU'),
                    velocity_output_mode=velocity_output_mode,
                    velocity_target=('difference_of_predictor_to_final_corrections'
                                     if config.get('velocity_var_to_predict') == 'ddUCorr'
                                     else 'difference_of_timestep_velocity_increments'
                                     if config.get('velocity_var_to_predict', 'dU') == 'ddU'
                                     else 'U_converged_minus_U_start') if joint else None)
    if args.resume or args.evaluate_only:
        previous_metadata = json.loads((data_dir / 'graph_model.json').read_text())
        for key in ('signature', 'feature_names', 'var_to_predict', 'levels', 'width', 'steps',
                    'channels', 'predict_dU_output', 'velocity_var_to_predict', 'velocity_output_mode'):
            if previous_metadata.get(key) != metadata[key]:
                raise ValueError(f'Cannot reuse mesh-graph checkpoint: {key} changed.')
        with np.load(data_dir / 'graph_stats.npz') as stats:
            mean, std = stats['mean'], stats['std']
            target_scale = float(stats['target_scale'])
            velocity_scale = float(stats['velocity_scale']) if joint else 1.
        velocity_shift_length_scale = float(previous_metadata.get('velocity_shift_length_scale', velocity_shift_length_scale))
        if joint and velocity_output_mode == 'shift_source':
            velocity_gradients = [coarse_vector_gradient(previous / velocity_scale, graph).astype(np.float32)
                                  for previous in previous_velocity_coarse]
        if args.resume:
            metadata['retraining_updates'] = previous_metadata.get('retraining_updates', 0) + 1
            print(f'[MESH GRAPH] resuming weights with fixed normalization; latest {len(samples)} samples', flush=True)
    else:
        np.savez(data_dir / 'graph_stats.npz', mean=mean, std=std, target_scale=target_scale, velocity_scale=velocity_scale)
    if args.evaluate_only:
        from .graph_model import MeshGraphShifter
        model = MeshGraphShifter(graph, len(mean), metadata['width'], metadata['steps'], predict_velocity=joint,
                                 velocity_output_mode=velocity_output_mode)
        model.load_weights(str(data_dir / 'graph_weights.h5'))
        return evaluate_validation(model, validation, sample_keys[split:], graph,
            dict(mean=mean, std=std, target_scale=target_scale, velocity_scale=velocity_scale,
                 velocity_shift_length_scale=velocity_shift_length_scale), config, source,
            args.plot_dir or data_dir / 'training_plots' / config.get('var_to_predict', 'ddp'))
    if joint and velocity_output_mode == 'shift_source':
        metadata['velocity_shift_length_scale'] = velocity_shift_length_scale
    metadata['graph_loss'] = config.get('graph_loss', 'coarse_mse')
    metadata['coarse_volume_weight_power'] = float(config.get('graph_volume_weight_power', 1.0))
    metadata['epochs_requested'] = config.get('num_epochs', 200)
    started_utc = datetime.now(timezone.utc).isoformat()
    plot_root = data_dir / 'training_plots' / config.get('var_to_predict', 'ddp')
    if args.resume:
        update_id = int(metadata['retraining_updates'])
        plot_directory = plot_root / f'retrain_{update_id:03d}'
    else:
        update_id = 0
        plot_directory = plot_root
    plot_directory.mkdir(parents=True, exist_ok=True)
    if args.resume:
        run_record = dict(
            update=update_id, status='running', started_utc=started_utc, completed_utc='',
            samples=json.dumps(sample_keys), training_samples=json.dumps(sample_keys[:split]),
            validation_samples=json.dumps(sample_keys[split:]),
            features=json.dumps(feature_names(config)), pressure_target=config.get('var_to_predict', 'ddp'),
            velocity_target=config.get('velocity_var_to_predict', 'dU') if joint else '',
            epochs_requested=config.get('num_epochs', 200), learning_rate=config.get('lr', 5e-4),
            pre_update_validation=None, best_validation=None, checkpoint_updated=None,
            output_directory=str(plot_directory.relative_to(data_dir)))
        (plot_directory / 'run.json').write_text(json.dumps(run_record, indent=2))
        update_retraining_index(plot_root / 'retraining_index.csv', run_record)
    if native_enabled:
        from .native_loss import NativeStateLoss, blend_coarse_and_native_loss, native_gradient_scales
        gradient_scales = native_gradient_scales(native_samples[:split], native_geometry, config, target_scale, velocity_scale)
        native_criterion = NativeStateLoss(graph, native_geometry, config, target_scale, velocity_scale, gradient_scales)
        metadata['native_loss'] = dict(**native_geometry['diagnostics'],
            pressure_gradient_scale=gradient_scales[0], velocity_gradient_scale=gradient_scales[1],
            pressure_gradient_weight=float(config.get('loss_pressure_gradient_weight', .2)),
            velocity_gradient_weight=float(config.get('loss_velocity_gradient_weight', .1)),
            operator='native internal-face owner-neighbour directional differences, not OpenFOAM FV residuals')
        if graph_loss == 'coarse_native_blend':
            metadata['native_loss']['blend_fraction'] = float(config.get('graph_native_loss_fraction', .25))
    (data_dir / 'graph_model.json').write_text(json.dumps(metadata, indent=2))
    print(f'[MESH GRAPH] chronological split: {split} train, {len(validation)} validation graphs', flush=True)

    import tensorflow as tf
    from .graph_model import MeshGraphShifter, reconstruct, reconstruct_velocity_shift

    tf.keras.utils.set_random_seed(42)
    model = MeshGraphShifter(graph, len(mean), metadata['width'], metadata['steps'], predict_velocity=joint,
                             velocity_output_mode=velocity_output_mode)
    if args.resume:
        model.load_weights(str(data_dir / 'graph_weights.h5'))
    optimizer = tf.keras.optimizers.Adam(float(config.get('lr', 5e-4)))
    optimizer.build(model.trainable_variables)
    gauge_weights = tf.constant(
        coarse_volume_weights(graph['coarse_volumes']), dtype=tf.float32
    )
    loss_weights = tf.constant(
        coarse_volume_weights(graph['coarse_volumes'], metadata['coarse_volume_weight_power']),
        dtype=tf.float32
    )

    @tf.function
    def step(features, gradient, label, training, native_sample=None, velocity_gradient=None):
        with tf.GradientTape() as tape:
            output = model((features - mean) / std, training=training)
            prediction = reconstruct(output, gradient, target_scale)
            prediction -= tf.reduce_sum(prediction * gauge_weights)
            pressure_label = label[:, 0] if joint else label
            pressure_loss = tf.reduce_sum(tf.square((prediction - pressure_label) / target_scale) * loss_weights)
            velocity_loss = tf.constant(0., dtype=tf.float32)
            if joint:
                velocity_prediction = (reconstruct_velocity_shift(output, velocity_gradient,
                                        velocity_shift_length_scale)
                                       if velocity_output_mode == 'shift_source' else output[:, 4:7])
                velocity_error = velocity_prediction - label[:, 1:4] / velocity_scale
                velocity_loss = tf.reduce_sum(tf.reduce_mean(tf.square(velocity_error), axis=1) * loss_weights)
            loss = pressure_loss + float(config.get('velocity_loss_weight', 1.)) * velocity_loss
            coarse_parts = tf.stack((loss, pressure_loss, velocity_loss))
            if native_enabled:
                native_parts = native_criterion(output, gradient, native_sample)
                if graph_loss == 'coarse_native_blend':
                    parts = blend_coarse_and_native_loss(
                        coarse_parts, native_parts, config.get('graph_native_loss_fraction', .25))
                else:
                    parts = native_parts
                loss, pressure_loss, velocity_loss = parts[0], parts[1], parts[2]
            else:
                parts = coarse_parts
        if training:
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return parts

    best = float('inf')
    pre_update_validation = None
    if args.resume:
        best = float(np.mean([step(*sample, False, native_samples[split + index] if native_enabled else None,
                        velocity_gradients[split + index] if velocity_output_mode == 'shift_source' else None).numpy()
                              for index, sample in enumerate(validation)], axis=0)[0])
        if not np.isfinite(best):
            raise FloatingPointError('Nonfinite validation loss in existing mesh-graph model.')
        pre_update_validation = best
        print(f'[MESH GRAPH] pre-update validation={best:.6g}', flush=True)
    stale = 0
    history = []
    component_history = []
    joint_loss_history = []
    joint_loss_history = []
    rng = np.random.default_rng(42)
    for epoch in range(int(config.get('num_epochs', 200))):
        losses = np.asarray([step(*training[index], True, native_samples[index] if native_enabled else None,
                      velocity_gradients[index] if velocity_output_mode == 'shift_source' else None).numpy()
                     for index in rng.permutation(len(training))])
        val_parts = np.mean([step(*sample, False, native_samples[split + index] if native_enabled else None,
                      velocity_gradients[split + index] if velocity_output_mode == 'shift_source' else None).numpy()
                     for index, sample in enumerate(validation)], axis=0)
        val_loss = float(val_parts[0])
        if not np.isfinite(losses).all() or not np.isfinite(val_loss):
            raise FloatingPointError('Nonfinite mesh-graph loss; surrogate activation aborted.')
        message = f'[MESH GRAPH] epoch={epoch + 1} train={np.mean(losses[:, 0]):.6g} validation={val_loss:.6g}'
        if joint:
            message += f' pressure={val_parts[1]:.6g} velocity={val_parts[2]:.6g}'
            joint_loss_history.append((epoch + 1, float(np.mean(losses[:, 1])), float(val_parts[1]),
                                       float(np.mean(losses[:, 2])), float(val_parts[2])))
            from .plots import plot_joint_loss_history
            plot_joint_loss_history(joint_loss_history, plot_directory,
                                    make_plot=config.get('plot_training', True) or args.resume)
        if native_enabled:
            message += f' pressure_gradient={val_parts[3]:.6g} velocity_gradient={val_parts[4]:.6g}'
            component_history.append((epoch + 1, *np.mean(losses, axis=0).tolist(), *val_parts.tolist()))
            from .plots import plot_loss_components
            plot_loss_components(component_history, plot_directory,
                                 make_plot=config.get('plot_training', True) or args.resume)
        print(message, flush=True)
        history.append((epoch + 1, float(np.mean(losses[:, 0])), val_loss))
        if config.get('plot_training', True) or args.resume:
            from .plots import plot_loss
            plot_loss(history, plot_directory)
        if val_loss < best:
            best, stale = val_loss, 0
            model.save_weights(str(data_dir / 'graph_weights.h5'))
        else:
            stale += 1
        if stale >= int(config.get('graph_patience', 20)):
            break
    model.load_weights(str(data_dir / 'graph_weights.h5'))
    if native_enabled:
        from .native_loss import evaluate_native_state
        evaluate_native_state(model, validation, native_samples[split:], sample_keys[split:], native_criterion,
                              native_geometry, dict(mean=mean, std=std), plot_directory)
    evaluate_validation(model, validation, sample_keys[split:], graph,
                        dict(mean=mean, std=std, target_scale=target_scale, velocity_scale=velocity_scale,
                             velocity_shift_length_scale=velocity_shift_length_scale),
                        config, source, plot_directory)
    if args.resume:
        completed_utc = datetime.now(timezone.utc).isoformat()
        run_record.update(status='completed', completed_utc=completed_utc,
                          pre_update_validation=pre_update_validation, best_validation=best,
                          checkpoint_updated=best < pre_update_validation)
        (plot_directory / 'run.json').write_text(json.dumps(run_record, indent=2))
        update_retraining_index(plot_root / 'retraining_index.csv', run_record)
    print(f'[MESH GRAPH] training complete; best validation={best:.6g}; original data.h5 preserved', flush=True)


if __name__ == '__main__':
    main()