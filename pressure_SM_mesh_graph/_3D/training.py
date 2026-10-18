import argparse
import csv
import importlib
import json
from pathlib import Path

import h5py
import numpy as np

from .graph_io import (feature_names, features_from_fields, load_mesh_graph, target_from_fields,
                       validate_config, velocity_target_from_fields, compose_velocity_increment)
from .mesh_graph import restrict


def evaluate_validation(model, samples, sample_keys, graph, stats, config, source, plot_directory):
    from .graph_model import reconstruct
    from .plots import (comparison_metrics, summarize_comparisons, print_comparison_summary,
                        plot_pressure_comparison, plot_prediction_statistics, plot_shifter_components,
                        plot_validation_history)
    plot_directory = Path(plot_directory)
    plot_directory.mkdir(parents=True, exist_ok=True)
    mode = config.get('var_to_predict', 'ddp')
    joint = config.get('predict_dU_output', False)
    records = []
    with h5py.File(source, 'r') as handle:
        for index, ((features, gradient, label), name) in enumerate(zip(samples, sample_keys)):
            group = handle[name]
            scale = max(float(np.asarray(group['U_MAX_NORM'][()]).reshape(-1)[0]), 1e-12)
            previous = restrict(group['dpPrev'][:].reshape(-1), graph['mapping'], graph['volumes'], graph['coarse_count'])
            output = model((features - stats['mean']) / stats['std'], training=False)
            predicted = reconstruct(output, gradient, stats['target_scale']).numpy().astype(float)
            predicted -= np.average(predicted, weights=graph['coarse_volumes'])
            pressure_scale = scale ** 2 * float(config.get('output_weight_factor', 1.))
            predicted *= pressure_scale
            pressure_label = label[:, 0] if joint else label
            truth = pressure_label.astype(float) * scale ** 2
            if mode == 'ddp':
                predicted += float(config.get('extrapolation_weight_factor', 1.)) * previous
                truth += previous
            for field in (predicted, previous, truth):
                field -= np.average(field, weights=graph['coarse_volumes'])
            metrics = comparison_metrics(predicted, previous, truth, graph['coarse_volumes'], mode)
            if joint:
                velocity = output[:, 4:7].numpy() * stats['velocity_scale'] * scale
                velocity *= float(config.get('velocity_output_weight_factor', config.get('output_weight_factor', 1.)))
                velocity_truth = label[:, 1:4] * scale
                if 'dUPrev' in group:
                    previous_velocity = restrict(group['dUPrev'][:], graph['mapping'], graph['volumes'], graph['coarse_count'])
                    velocity = compose_velocity_increment(velocity, previous_velocity, config)
                    if config.get('velocity_var_to_predict', 'dU') == 'ddU':
                        velocity_truth += previous_velocity
                    metrics['velocity_trend_rmse'] = float(np.sqrt(np.average(np.sum((previous_velocity - velocity_truth) ** 2, axis=1),
                        weights=graph['coarse_volumes'])))
                metrics['velocity_rmse'] = float(np.sqrt(np.average(np.sum((velocity - velocity_truth) ** 2, axis=1),
                    weights=graph['coarse_volumes'])))
                metrics['velocity_start_rmse'] = float(np.sqrt(np.average(np.sum(velocity_truth ** 2, axis=1),
                    weights=graph['coarse_volumes'])))
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
                plot_shifter_components(graph['coarse_centres'], output.numpy(), gradient, stats['target_scale'],
                                        pressure_scale, plot_directory, plot_name)
                if joint:
                    from .plots import plot_velocity_comparison
                    plot_velocity_comparison(graph['coarse_centres'], velocity, velocity_truth, plot_directory, plot_name)
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
    parser.add_argument('--data_dir', default='ML_data')
    args = parser.parse_args(argv)
    config = vars(importlib.import_module('python_module'))
    validate_config(config)
    joint = config.get('predict_dU_output', False)
    data_dir = Path(args.data_dir)
    source = data_dir / 'data.h5'
    with h5py.File(source, 'r') as handle:
        required = ('coordinates', 'cell_volumes', 'mesh_bounds')
        if any(name not in handle for name in required):
            raise ValueError('Fresh solver samples with coordinates, volumes and mesh bounds are required.')
        centres, volumes = handle['coordinates'][:], handle['cell_volumes'][:]
        graph, signature = load_mesh_graph(centres, volumes, Path.cwd(), data_dir,
                                           int(config.get('graph_levels', 3)))
        sample_keys = sorted((name for name in handle if name.startswith('sample_')),
                             key=lambda name: float(handle[name].attrs.get('timestep', name.split('_')[-1])))
        if len(sample_keys) < 3:
            raise ValueError('At least three timestep samples are required for graph training/validation.')
        samples = []
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
                velocity_fields = {'dU': group['dU'][:]}
                if 'dUPrev' in group:
                    velocity_fields['dUPrev'] = group['dUPrev'][:]
                velocity_label = restrict(velocity_target_from_fields(velocity_fields, config) / max(scale, 1e-12),
                                          graph['mapping'], volumes, graph['coarse_count'])
                if velocity_label.shape != (graph['coarse_count'], 3) or not np.isfinite(velocity_label).all():
                    raise ValueError(f'Invalid dU target in {name}.')
                label = np.column_stack((label, velocity_label))
            samples.append((features, gradient, label.astype(np.float32)))
    split = max(1, min(len(samples) - 1, int(.8 * len(samples))))
    training, validation = samples[:split], samples[split:]
    mean = np.mean([features.mean(axis=0) for features, _, _ in training], axis=0)
    variance = np.mean([np.mean((features - mean) ** 2, axis=0) for features, _, _ in training], axis=0)
    std = np.maximum(np.sqrt(variance), 1e-7)
    target_scale = max(float(np.std(np.concatenate([label[:, 0] if joint else label for _, _, label in training]))), 1e-7)
    velocity_scale = max(float(np.sqrt(np.mean(np.concatenate([label[:, 1:4] for _, _, label in training]) ** 2))), 1e-7) if joint else 1.
    np.savez(data_dir / 'graph_stats.npz', mean=mean, std=std, target_scale=target_scale, velocity_scale=velocity_scale)
    metadata = dict(signature=signature, feature_names=feature_names(config),
                    var_to_predict=config.get('var_to_predict', 'ddp'), levels=int(config.get('graph_levels', 3)),
                    width=int(config.get('graph_width', 24)), steps=int(config.get('graph_steps', 4)),
                    channels=len(mean), samples=len(samples), training=split, validation=len(validation),
                    predict_dU_output=bool(joint), velocity_var_to_predict=config.get('velocity_var_to_predict', 'dU'),
                    velocity_target=('difference_of_timestep_velocity_increments'
                                     if config.get('velocity_var_to_predict', 'dU') == 'ddU'
                                     else 'U_converged_minus_U_start') if joint else None)
    (data_dir / 'graph_model.json').write_text(json.dumps(metadata, indent=2))
    print(f'[MESH GRAPH] chronological split: {split} train, {len(validation)} validation graphs', flush=True)

    import tensorflow as tf
    from .graph_model import MeshGraphShifter, reconstruct

    tf.keras.utils.set_random_seed(42)
    model = MeshGraphShifter(graph, len(mean), metadata['width'], metadata['steps'], predict_velocity=joint)
    optimizer = tf.keras.optimizers.Adam(float(config.get('lr', 5e-4)))
    node_weights = tf.constant(graph['coarse_volumes'] / graph['coarse_volumes'].sum(), dtype=tf.float32)

    @tf.function
    def step(features, gradient, label, training):
        with tf.GradientTape() as tape:
            output = model((features - mean) / std, training=training)
            prediction = reconstruct(output, gradient, target_scale)
            prediction -= tf.reduce_sum(prediction * node_weights)
            pressure_label = label[:, 0] if joint else label
            pressure_loss = tf.reduce_sum(tf.square((prediction - pressure_label) / target_scale) * node_weights)
            velocity_loss = tf.constant(0., dtype=tf.float32)
            if joint:
                velocity_error = output[:, 4:7] - label[:, 1:4] / velocity_scale
                velocity_loss = tf.reduce_sum(tf.reduce_mean(tf.square(velocity_error), axis=1) * node_weights)
            loss = pressure_loss + float(config.get('velocity_loss_weight', 1.)) * velocity_loss
        if training:
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return tf.stack((loss, pressure_loss, velocity_loss))

    best = float('inf')
    stale = 0
    history = []
    plot_directory = data_dir / 'training_plots' / config.get('var_to_predict', 'ddp')
    rng = np.random.default_rng(42)
    for epoch in range(int(config.get('num_epochs', 200))):
        losses = np.asarray([step(*training[index], True).numpy() for index in rng.permutation(len(training))])
        val_parts = np.mean([step(*sample, False).numpy() for sample in validation], axis=0)
        val_loss = float(val_parts[0])
        if not np.isfinite(losses).all() or not np.isfinite(val_loss):
            raise FloatingPointError('Nonfinite mesh-graph loss; surrogate activation aborted.')
        message = f'[MESH GRAPH] epoch={epoch + 1} train={np.mean(losses[:, 0]):.6g} validation={val_loss:.6g}'
        if joint:
            message += f' pressure={val_parts[1]:.6g} velocity={val_parts[2]:.6g}'
        print(message, flush=True)
        history.append((epoch + 1, float(np.mean(losses[:, 0])), val_loss))
        if config.get('plot_training', True):
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
    evaluate_validation(model, validation, sample_keys[split:], graph,
                        dict(mean=mean, std=std, target_scale=target_scale, velocity_scale=velocity_scale), config, source, plot_directory)
    print(f'[MESH GRAPH] training complete; best validation={best:.6g}; original data.h5 preserved', flush=True)


if __name__ == '__main__':
    main()