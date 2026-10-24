import numpy as np
import tensorflow as tf


class NativeProlongation:
    """Differentiable counterpart of the limited NumPy coarse-to-cell transfer."""

    def __init__(self, graph):
        from .mesh_graph import interface_geometry
        count = int(graph['coarse_count'])
        source, target = graph['source'], graph['target']
        offsets = graph['edge_offsets']
        weights = 1. / np.maximum(np.sum(offsets ** 2, axis=1), 1e-24)
        normal = np.zeros((count, 3, 3))
        np.add.at(normal, source, offsets[:, :, None] * offsets[:, None, :] * weights[:, None, None])
        self.inverse = tf.constant(np.linalg.pinv(normal, rcond=1e-10, hermitian=True), tf.float64)
        self.source = tf.constant(source, tf.int32)
        self.target = tf.constant(target, tf.int32)
        self.edge_vectors = tf.constant(offsets * weights[:, None], tf.float64)
        self.mapping = tf.constant(graph['mapping'], tf.int32)
        self.displacement = tf.constant(graph['centres'] - graph['coarse_centres'][graph['mapping']], tf.float64)
        coarse_volumes = np.bincount(graph['mapping'], weights=graph['volumes'], minlength=count)
        self.relative_volumes = tf.constant(graph['volumes'] / coarse_volumes[graph['mapping']], tf.float64)
        owner, neighbour, offsets, weights, step = interface_geometry(graph)
        self.interface_owner = tf.constant(owner, tf.int32)
        self.interface_neighbour = tf.constant(neighbour, tf.int32)
        self.interface_offsets = tf.constant(offsets, tf.float64)
        self.interface_weights = tf.constant(weights, tf.float64)
        self.interface_step = tf.constant(step, tf.float64)
        self.volumes = tf.constant(graph['volumes'], tf.float64)
        self.cell_count = len(graph['mapping'])
        self.has_interfaces = bool(len(owner))
        self.count = count

    def __call__(self, values, linear=True):
        values = tf.cast(tf.convert_to_tensor(values), tf.float64)
        result = tf.gather(values, self.mapping)
        if not linear:
            return tf.cast(result, tf.float32)
        difference = tf.gather(values, self.target) - tf.gather(values, self.source)
        rhs = tf.math.unsorted_segment_sum(self.edge_vectors * difference[:, None], self.source, self.count)
        gradient = tf.einsum('nij,nj->ni', self.inverse, rhs)
        correction = tf.reduce_sum(self.displacement * tf.gather(gradient, self.mapping), axis=1)
        average = tf.math.unsorted_segment_sum(correction * self.relative_volumes, self.mapping, self.count)
        correction -= tf.gather(average, self.mapping)
        lower = tf.minimum(values, tf.math.unsorted_segment_min(tf.gather(values, self.target), self.source, self.count))
        upper = tf.maximum(values, tf.math.unsorted_segment_max(tf.gather(values, self.target), self.source, self.count))
        minimum = tf.minimum(tf.math.unsorted_segment_min(correction, self.mapping, self.count), 0.)
        maximum = tf.maximum(tf.math.unsorted_segment_max(correction, self.mapping, self.count), 0.)
        increasing = tf.where(maximum > 0, tf.math.divide_no_nan(upper - values, maximum), tf.ones_like(values))
        decreasing = tf.where(minimum < 0, tf.math.divide_no_nan(lower - values, minimum), tf.ones_like(values))
        factors = tf.clip_by_value(tf.minimum(tf.minimum(increasing, decreasing), 1.), 0., 1.)
        result += correction * tf.gather(factors, self.mapping)
        if self.has_interfaces:
            native_gradient = tf.gather(gradient, self.mapping)
            desired = tf.reduce_sum(self.interface_offsets * .5 * (
                tf.gather(native_gradient, self.interface_owner) +
                tf.gather(native_gradient, self.interface_neighbour)), axis=1)
            for _ in range(3):
                residual = self.interface_weights * (tf.gather(result, self.interface_neighbour)
                    - tf.gather(result, self.interface_owner) - desired)
                derivative = tf.math.unsorted_segment_sum(residual, self.interface_neighbour, self.cell_count)
                derivative -= tf.math.unsorted_segment_sum(residual, self.interface_owner, self.cell_count)
                update = -derivative / self.volumes
                average = tf.math.unsorted_segment_sum(update * self.relative_volumes, self.mapping, self.count)
                update -= tf.gather(average, self.mapping)
                update *= tf.gather(self.interface_step, self.mapping)
                increasing = tf.where(update > 0,
                    tf.math.divide_no_nan(tf.gather(upper, self.mapping) - result, update), tf.ones_like(update))
                decreasing = tf.where(update < 0,
                    tf.math.divide_no_nan(tf.gather(lower, self.mapping) - result, update), tf.ones_like(update))
                fraction = tf.clip_by_value(tf.minimum(increasing, decreasing), 0., 1.)
                factors = tf.math.unsorted_segment_min(fraction, self.mapping, self.count)
                result += update * tf.gather(factors, self.mapping)
        return tf.cast(result, tf.float32)


def native_loss_geometry(graph, owner, neighbour, config, obstacle_points=None):
    from scipy.spatial import cKDTree
    volumes = np.asarray(graph['volumes'], dtype=float)
    global_weights = volumes / volumes.sum()
    fraction = float(config.get('loss_obstacle_fraction', .5))
    radius = float(config.get('loss_obstacle_radius', .05))
    if not np.isfinite(fraction) or not 0 <= fraction <= 1 or not np.isfinite(radius) or radius <= 0:
        raise ValueError('Obstacle loss fraction must be in [0,1] and radius must be positive.')
    weights = global_weights.copy()
    roi = np.zeros(len(volumes), dtype=bool)
    if fraction > 0:
        if obstacle_points is None or len(obstacle_points) == 0:
            raise ValueError('Obstacle-focused native loss requires a matching sampled boundary patch.')
        distance = cKDTree(obstacle_points).query(graph['centres'])[0]
        roi = distance <= radius
        if not roi.any():
            raise ValueError('No cells inside loss_obstacle_radius; increase the radius for this mesh.')
        focused = volumes * np.exp(-.5 * (distance / radius) ** 2)
        focused /= focused.sum()
        weights = (1 - fraction) * global_weights + fraction * focused
    owner, neighbour = np.asarray(owner), np.asarray(neighbour)
    distance = np.linalg.norm(graph['centres'][owner] - graph['centres'][neighbour], axis=1)
    if not len(distance) or np.any(distance <= 0) or not np.isfinite(distance).all():
        raise ValueError('Native gradient loss requires positive finite internal-face cell-centre distances.')
    edge_weights = weights[owner] + weights[neighbour]
    edge_weights /= edge_weights.sum()
    return dict(weights=weights, global_weights=global_weights, roi=roi,
                owner=owner, neighbour=neighbour, distance=distance, edge_weights=edge_weights,
                diagnostics=dict(obstacle_fraction=fraction, obstacle_radius=radius,
                                 roi_cells=int(roi.sum()), roi_volume_fraction=float(global_weights[roi].sum()),
                                 roi_loss_fraction=float(weights[roi].sum()),
                                 distance_definition='nearest sampled obstacle-face centre'))


def native_gradient_scales(samples, geometry, config, pressure_scale, velocity_scale):
    owner, neighbour, distance = geometry['owner'], geometry['neighbour'], geometry['distance']
    pressure_means, velocity_means = [], []
    for sample in samples:
        pressure = sample['pressure']
        if config.get('var_to_predict', 'ddp') == 'ddp':
            pressure = pressure - sample['pressure_previous']
        pressure_gradient = (pressure[owner] - pressure[neighbour]) / distance
        pressure_means.append(np.sum(geometry['edge_weights'] * pressure_gradient ** 2))
        if config.get('predict_dU_output', False):
            velocity = sample['velocity']
            if config.get('velocity_var_to_predict', 'dU') == 'ddU':
                velocity = velocity - sample['velocity_previous']
            velocity_gradient = (velocity[owner] - velocity[neighbour]) / distance[:, None]
            velocity_means.append(np.sum(geometry['edge_weights'] * np.mean(velocity_gradient ** 2, axis=1)))
    length = max(float(np.median(distance)), 1e-12)
    pressure = max(float(np.sqrt(np.mean(pressure_means))), pressure_scale / length * 1e-3, 1e-7)
    velocity = max(float(np.sqrt(np.mean(velocity_means))) if velocity_means else 0., velocity_scale / length * 1e-3, 1e-7)
    return pressure, velocity


class NativeStateLoss:
    """Value and directional-gradient errors after inference-equivalent native transfer."""

    def __init__(self, graph, geometry, config, pressure_scale, velocity_scale, gradient_scales):
        self.transfer = NativeProlongation(graph)
        self.config = dict(config)
        self.pressure_scale = float(pressure_scale)
        self.velocity_scale = float(velocity_scale)
        self.gradient_scales = gradient_scales
        self.weights = tf.constant(geometry['weights'], tf.float32)
        self.global_weights = tf.constant(geometry['global_weights'], tf.float32)
        self.coarse_weights = tf.constant(graph['coarse_volumes'] / graph['coarse_volumes'].sum(), tf.float32)
        self.owner = tf.constant(geometry['owner'], tf.int32)
        self.neighbour = tf.constant(geometry['neighbour'], tf.int32)
        self.distance = tf.constant(geometry['distance'], tf.float32)
        self.edge_weights = tf.constant(geometry['edge_weights'], tf.float32)
        self.mapping = tf.constant(graph['mapping'], tf.int32)
        self.relative_volumes = tf.cast(self.transfer.relative_volumes, tf.float32)
        self.count = graph['coarse_count']

    def gradient_error(self, error, scale):
        difference = tf.gather(error, self.owner) - tf.gather(error, self.neighbour)
        denominator = self.distance if error.shape.rank == 1 else self.distance[:, None]
        squared = tf.square(difference / denominator / scale)
        if error.shape.rank != 1:
            squared = tf.reduce_mean(squared, axis=1)
        return tf.reduce_sum(squared * self.edge_weights)

    def __call__(self, output, gradient, sample, return_errors=False):
        from .graph_model import reconstruct
        config = self.config
        pressure = reconstruct(output, gradient, self.pressure_scale)
        pressure -= tf.reduce_sum(pressure * self.coarse_weights)
        pressure *= float(config.get('output_weight_factor', 1.))
        previous = sample['pressure_previous']
        residual_mode = config.get('var_to_predict', 'ddp') == 'ddp'
        full = residual_mode and config.get('ddp_full_coarse_prediction', False)
        if full:
            coarse_previous = tf.math.unsorted_segment_sum(previous * self.relative_volumes, self.mapping, self.count)
            pressure += float(config.get('extrapolation_weight_factor', 1.)) * coarse_previous
        pressure = self.transfer(pressure, bool(config.get('graph_linear_prolongation', True)))
        if residual_mode and not full:
            pressure += float(config.get('extrapolation_weight_factor', 1.)) * previous
        pressure_error = pressure - sample['pressure']
        pressure_error -= tf.reduce_sum(pressure_error * self.global_weights)
        pressure_loss = tf.reduce_sum(tf.square(pressure_error / self.pressure_scale) * self.weights)
        pressure_gradient = self.gradient_error(pressure_error, self.gradient_scales[0])
        velocity_loss, velocity_gradient = tf.constant(0., tf.float32), tf.constant(0., tf.float32)
        velocity_error = tf.zeros((tf.shape(pressure_error)[0], 3), tf.float32)
        if config.get('predict_dU_output', False):
            velocity = output[:, 4:7] * self.velocity_scale
            velocity *= float(config.get('velocity_output_weight_factor', config.get('output_weight_factor', 1.)))
            velocity = tf.stack([self.transfer(velocity[:, component], bool(config.get('velocity_linear_prolongation', False)))
                                 for component in range(3)], axis=1)
            if config.get('velocity_var_to_predict', 'dU') == 'ddU':
                velocity += float(config.get('velocity_extrapolation_weight_factor', 1.)) * sample['velocity_previous']
            velocity_error = velocity - sample['velocity']
            velocity_loss = tf.reduce_sum(tf.reduce_mean(tf.square(velocity_error / self.velocity_scale), axis=1) * self.weights)
            velocity_gradient = self.gradient_error(velocity_error, self.gradient_scales[1])
        total = (pressure_loss + float(config.get('velocity_loss_weight', 1.)) * velocity_loss
                 + float(config.get('loss_pressure_gradient_weight', .2)) * pressure_gradient
                 + float(config.get('loss_velocity_gradient_weight', .1)) * velocity_gradient)
        parts = tf.stack((total, pressure_loss, velocity_loss, pressure_gradient, velocity_gradient))
        return (parts, pressure_error, velocity_error) if return_errors else parts


def blend_coarse_and_native_loss(coarse_parts, native_parts, native_fraction):
    native_fraction = float(native_fraction)
    if not np.isfinite(native_fraction) or not 0 <= native_fraction <= 1:
        raise ValueError('native_fraction must be finite and in [0,1].')
    coarse_fraction = 1. - native_fraction
    return tf.stack((
        coarse_fraction * coarse_parts[0] + native_fraction * native_parts[0],
        coarse_fraction * coarse_parts[1] + native_fraction * native_parts[1],
        coarse_fraction * coarse_parts[2] + native_fraction * native_parts[2],
        native_fraction * native_parts[3],
        native_fraction * native_parts[4],
    ))


def evaluate_native_state(model, samples, native_samples, names, criterion, geometry, stats, directory):
    import csv
    import json
    from pathlib import Path
    records = []
    owner, neighbour, distance = geometry['owner'], geometry['neighbour'], geometry['distance']
    edge_weights = geometry['global_weights'][owner] + geometry['global_weights'][neighbour]
    edge_weights /= edge_weights.sum()
    roi = geometry['roi']

    def scalar_rms(error, selected=None):
        weights = geometry['global_weights']
        if selected is not None:
            if not selected.any():
                return None
            error, weights = error[selected], weights[selected]
        return float(np.sqrt(np.average(error ** 2, weights=weights)))

    for (features, gradient, _), native, name in zip(samples, native_samples, names):
        output = model((features - stats['mean']) / stats['std'], training=False)
        parts, pressure_error, velocity_error = criterion(output, gradient, native, return_errors=True)
        scale = float(native['scale'])
        pressure_error = pressure_error.numpy().astype(float) * scale ** 2
        velocity_error = velocity_error.numpy().astype(float) * scale
        pressure_gradient = (pressure_error[owner] - pressure_error[neighbour]) / distance
        velocity_gradient = (velocity_error[owner] - velocity_error[neighbour]) / distance[:, None]
        previous_pressure = (native['pressure_previous'] - native['pressure']).astype(float) * scale ** 2
        previous_pressure -= np.average(previous_pressure, weights=geometry['global_weights'])
        velocity_previous = (native['velocity_previous'] - native['velocity']).astype(float) * scale
        record = dict(sample=name, pressure_rmse=scalar_rms(pressure_error), pressure_obstacle_rmse=scalar_rms(pressure_error, roi),
            pressure_trend_rmse=scalar_rms(previous_pressure),
            pressure_edge_gradient_rmse=float(np.sqrt(np.sum(pressure_gradient ** 2 * edge_weights))),
            pressure_gradient_obstacle_rmse=None, velocity_rmse=scalar_rms(np.linalg.norm(velocity_error, axis=1)),
            velocity_obstacle_rmse=scalar_rms(np.linalg.norm(velocity_error, axis=1), roi),
            velocity_trend_rmse=scalar_rms(np.linalg.norm(velocity_previous, axis=1)),
            velocity_edge_gradient_rmse=float(np.sqrt(np.sum(np.sum(velocity_gradient ** 2, axis=1) * edge_weights))))
        roi_edges = roi[owner] | roi[neighbour]
        if roi_edges.any():
            record['pressure_gradient_obstacle_rmse'] = float(np.sqrt(np.average(pressure_gradient[roi_edges] ** 2,
                weights=edge_weights[roi_edges])))
        if not criterion.config.get('predict_dU_output', False):
            for key in ('velocity_rmse', 'velocity_obstacle_rmse', 'velocity_trend_rmse', 'velocity_edge_gradient_rmse'):
                record[key] = None
        records.append(record)
    summary = {key: float(np.sqrt(np.mean([row[key] ** 2 for row in records])))
               if records[0][key] is not None else None for key in records[0] if key != 'sample'}
    summary.update(samples=len(records), metrics_scope='native cells after Python prolongation/baseline composition, before OpenFOAM boundary correction/reconstruction',
                   pressure_gauge='volume_mean_free', gradient_operator='internal-face owner-neighbour directional difference',
                   roi=geometry['diagnostics'])
    directory = Path(directory)
    with (directory / 'native_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (directory / 'native_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print('[MESH GRAPH] NATIVE-INJECTION SUMMARY: ' + json.dumps(summary), flush=True)
    return summary