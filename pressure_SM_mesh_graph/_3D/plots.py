import csv
from pathlib import Path

import numpy as np


def _figure(rows, columns):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(4 * columns, 2.8 * rows), constrained_layout=True)
    FigureCanvasAgg(figure)
    return figure, figure.subplots(rows, columns, squeeze=False)


def plot_loss(history, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'loss.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('epoch', 'training_loss', 'validation_loss'))
        writer.writerows(history)
    values = np.asarray(history)
    figure, axes = _figure(1, 1)
    axes[0, 0].plot(values[:, 0], values[:, 1], label='Training')
    axes[0, 0].plot(values[:, 0], values[:, 2], label='Validation')
    if np.all(values[:, 1:] > 0):
        axes[0, 0].set_yscale('log')
    axes[0, 0].set(xlabel='Epoch', ylabel='Volume-weighted normalized MSE', title='Whole-domain graph loss')
    axes[0, 0].legend()
    figure.savefig(directory / 'loss.png', dpi=130)


def plot_joint_loss_history(history, directory, make_plot=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'loss_pressure_velocity.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('epoch', 'training_pressure', 'validation_pressure',
                         'training_velocity', 'validation_velocity'))
        writer.writerows(history)
    if not make_plot:
        return
    values = np.asarray(history, dtype=float)
    figure, axes = _figure(1, 2)
    for axis, train_column, validation_column, title in (
            (axes[0, 0], 1, 2, 'Pressure loss'), (axes[0, 1], 3, 4, 'Velocity loss')):
        axis.plot(values[:, 0], values[:, train_column], label='Training')
        axis.plot(values[:, 0], values[:, validation_column], label='Validation')
        if np.all(values[:, 1:] > 0):
            axis.set_yscale('log')
        axis.set(title=title, xlabel='Epoch', ylabel='Volume-weighted normalized MSE')
        axis.legend()
    figure.suptitle('Joint graph model loss components')
    figure.savefig(directory / 'loss_pressure_velocity.png', dpi=130)


def plot_loss_components(history, directory, make_plot=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = ('total', 'pressure', 'velocity', 'pressure_gradient', 'velocity_gradient')
    columns = ['epoch'] + [f'training_{name}' for name in names] + [f'validation_{name}' for name in names]
    with (directory / 'loss_components.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(history)
    if not make_plot:
        return
    values = np.asarray(history, dtype=float)
    figure, axes = _figure(2, 2)
    groups = (
        (axes[0, 0], 'Total loss', (1, 6)),
        (axes[0, 1], 'Pressure loss', (2, 7)),
        (axes[1, 0], 'Velocity loss', (3, 8)),
        (axes[1, 1], 'Gradient losses', (4, 9, 5, 10)),
    )
    for axis, title, indexes in groups:
        for train_index, validation_index in zip(indexes[::2], indexes[1::2]):
            suffix = '' if len(indexes) == 2 else ('pressure gradient' if train_index == 4 else 'velocity gradient')
            axis.plot(values[:, 0], values[:, train_index], label=f'Training {suffix}'.strip())
            axis.plot(values[:, 0], values[:, validation_index], linestyle='--',
                      label=f'Validation {suffix}'.strip())
        if np.all(values[:, 1:] > 0):
            axis.set_yscale('log')
        axis.set(title=title, xlabel='Epoch', ylabel='Loss')
        axis.legend()
    figure.suptitle('Native-state graph loss components')
    figure.savefig(directory / 'loss_components.png', dpi=130)


def plot_joint_loss_history(history, directory, make_plot=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'loss_pressure_velocity.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('epoch', 'training_pressure', 'validation_pressure',
                         'training_velocity', 'validation_velocity'))
        writer.writerows(history)
    if not make_plot:
        return
    values = np.asarray(history, dtype=float)
    figure, axes = _figure(1, 2)
    for axis, train_column, validation_column, title in (
            (axes[0, 0], 1, 2, 'Pressure loss'), (axes[0, 1], 3, 4, 'Velocity loss')):
        axis.plot(values[:, 0], values[:, train_column], label='Training')
        axis.plot(values[:, 0], values[:, validation_column], label='Validation')
        if np.all(values[:, 1:] > 0):
            axis.set_yscale('log')
        axis.set(title=title, xlabel='Epoch', ylabel='Volume-weighted normalized MSE')
        axis.legend()
    figure.suptitle('Joint graph model loss components')
    figure.savefig(directory / 'loss_pressure_velocity.png', dpi=130)


def pressure_components(returned, dp_previous, config):
    returned = np.asarray(returned).reshape(-1)
    previous = np.asarray(dp_previous).reshape(-1)
    if config.get('var_to_predict', 'ddp') == 'dp' or config.get('ddp_full_coarse_prediction', False):
        increment = returned.copy()
    else:
        increment = returned + float(config.get('extrapolation_weight_factor', 1.)) * previous
    return increment, increment - previous


def _improvement(error, baseline):
    return 100 * (1 - error / baseline) if baseline > 0 else float('nan')


def comparison_metrics(predicted_dp, previous_dp, truth_dp, volumes, prediction_mode='ddp'):
    weights = np.asarray(volumes).reshape(-1)
    weights = weights / weights.sum()
    predicted = np.asarray(predicted_dp).reshape(-1)
    previous = np.asarray(previous_dp).reshape(-1)
    truth = np.asarray(truth_dp).reshape(-1)
    error = float(np.sqrt(np.sum(weights * (predicted - truth) ** 2)))
    baseline = float(np.sqrt(np.sum(weights * (previous - truth) ** 2)))
    previous_error = float(np.sqrt(np.sum(weights * truth ** 2)))
    pxtrap_error = previous_error if prediction_mode == 'dp' else baseline
    mae = float(np.sum(weights * np.abs(predicted - truth)))
    previous_mae = float(np.sum(weights * np.abs(truth)))
    trend_mae = float(np.sum(weights * np.abs(previous - truth)))
    pxtrap_mae = previous_mae if prediction_mode == 'dp' else trend_mae
    return dict(sm_dp_rmse=error, extrapolation_dp_rmse=baseline,
                sm_ddp_rmse=error, extrapolation_ddp_rmse=baseline,
                improvement_percent=_improvement(error, baseline), p_prev_rmse=previous_error,
                pxtrap_rmse=pxtrap_error, improvement_vs_p_prev_percent=_improvement(error, previous_error),
                improvement_vs_pxtrap_percent=_improvement(error, pxtrap_error), sm_dp_mae=mae,
                p_prev_mae=previous_mae, extrapolation_dp_mae=trend_mae, pxtrap_mae=pxtrap_mae)


def summarize_comparisons(records, prediction_mode):
    if not records:
        raise ValueError('At least one held-out comparison is required.')
    summary = {}
    for name in ('sm_dp_rmse', 'p_prev_rmse', 'pxtrap_rmse', 'extrapolation_dp_rmse'):
        summary[name] = float(np.sqrt(np.mean([record[name] ** 2 for record in records])))
    for name in ('sm_dp_mae', 'p_prev_mae', 'pxtrap_mae', 'extrapolation_dp_mae'):
        summary[name] = float(np.mean([record[name] for record in records]))
    for label, key in (('p_prev', 'p_prev'), ('pxtrap', 'pxtrap'), ('trend', 'extrapolation_dp')):
        summary[f'improvement_vs_{label}_percent'] = _improvement(summary['sm_dp_rmse'], summary[f'{key}_rmse'])
    summary.update(samples=len(records), prediction_mode=prediction_mode,
                   gauge='volume_mean_free', representation='coarse_graph',
                   aggregation='equal_timestep_volume_weighted')
    return summary


def print_comparison_summary(summary):
    print(f"[MESH GRAPH] HELD-OUT SUMMARY: best checkpoint, {summary['samples']} timesteps, "
          f"{summary['representation']}, {summary['gauge']}, physical pressure units", flush=True)
    for label, key in (('p_prev', 'p_prev'), ('pExtrap', 'pxtrap'), ('p_prev + dpPrev', 'extrapolation_dp')):
        improvement = _improvement(summary['sm_dp_rmse'], summary[f'{key}_rmse'])
        change = f'{improvement:+.2f}%' if np.isfinite(improvement) else 'n/a (zero baseline error)'
        status = 'HELPED' if improvement > 0 else 'HURT' if improvement < 0 else 'UNCHANGED'
        if not np.isfinite(improvement):
            status = 'UNDEFINED'
        print(f"[MESH GRAPH] vs {label}: SM RMSE={summary['sm_dp_rmse']:.6g}, "
              f"baseline RMSE={summary[f'{key}_rmse']:.6g}, improvement={change} ({status}); "
              f"SM MAE={summary['sm_dp_mae']:.6g}, baseline MAE={summary[f'{key}_mae']:.6g}", flush=True)
    formula = 'p_prev' if summary['prediction_mode'] == 'dp' else 'p_prev + dpPrev'
    print(f'[MESH GRAPH] pExtrap baseline = {formula}; positive improvement means lower error.', flush=True)


def plot_pressure_comparison(centres, volumes, predicted_dp, previous_dp, directory, name,
                            truth_dp=None, previous_p=None, mean_free=False, max_points=30000,
                            append_metrics=True, slice_axis=2, slice_fraction=.5,
                            prediction_mode='ddp', write_metrics=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    centres = np.asarray(centres)
    volumes = np.asarray(volumes).reshape(-1)
    predicted_dp = np.asarray(predicted_dp, dtype=float).reshape(-1).copy()
    previous_dp = np.asarray(previous_dp, dtype=float).reshape(-1).copy()
    truth_dp = None if truth_dp is None else np.asarray(truth_dp, dtype=float).reshape(-1).copy()
    if mean_free:
        for field in (predicted_dp, previous_dp, truth_dp):
            if field is not None:
                field -= np.average(field, weights=volumes)
    rows = [('dp', predicted_dp, previous_dp, truth_dp),
            ('ddp', predicted_dp - previous_dp, np.zeros_like(previous_dp),
             None if truth_dp is None else truth_dp - previous_dp)]
    if previous_p is not None:
        previous_p = np.asarray(previous_p).reshape(-1)
        rows.insert(0, ('p', previous_p + predicted_dp, previous_p + previous_dp,
                        None if truth_dp is None else previous_p + truth_dp))
    middle = centres[:, slice_axis].min() + slice_fraction * np.ptp(centres[:, slice_axis])
    plane_axes = [axis for axis in range(3) if axis != slice_axis]
    width = max(.03 * np.ptp(centres[:, slice_axis]), 1e-12)
    selected = np.flatnonzero(np.abs(centres[:, slice_axis] - middle) <= width)
    if len(selected) < min(10, len(centres)):
        selected = np.argsort(np.abs(centres[:, slice_axis] - middle))[:min(2000, len(centres))]
    if len(selected) > max_points:
        selected = np.random.default_rng(42).choice(selected, max_points, replace=False)
    has_truth = truth_dp is not None
    figure, axes = _figure(len(rows), 6 if has_truth else 3)
    for row, (label, prediction, baseline, truth) in enumerate(rows):
        fields = [prediction, baseline]
        titles = ['SM', 'Extrapolation (dp = dpPrev)']
        if has_truth:
            previous_only = previous_p if label == 'p' else np.zeros_like(previous_dp) if label == 'dp' else -previous_dp
            fields = [truth] + fields + [prediction - truth, baseline - truth, previous_only - truth]
            titles = ['CFD target'] + titles + ['SM - CFD', 'Extrapolation - CFD', 'p_prev - CFD']
        else:
            fields.append(prediction - baseline)
            titles.append('SM - extrapolation')
        signal = np.concatenate([field[selected] for field in fields[:3 if has_truth else 2]])
        signal_limit = max(float(np.max(np.abs(signal))), 1e-12)
        error_limit = max(float(np.max(np.abs(np.concatenate([field[selected] for field in fields[-3 if has_truth else -1:]])))), 1e-12)
        for column, (field, title) in enumerate(zip(fields, titles)):
            axis = axes[row, column]
            limit = error_limit if column >= (3 if has_truth else 2) else signal_limit
            image = axis.scatter(centres[selected, plane_axes[0]], centres[selected, plane_axes[1]], c=field[selected],
                                 s=3, marker='s', cmap='RdBu_r', vmin=-limit, vmax=limit, linewidths=0)
            axis.set(title=f'{label}: {title}', xlabel='xyz'[plane_axes[0]], ylabel='xyz'[plane_axes[1]], aspect='equal')
            figure.colorbar(image, ax=axis, shrink=.75)
    status = 'CFD target available' if has_truth else 'Prediction-time view; CFD target not yet available'
    figure.suptitle(f'{name} | {"xyz"[slice_axis]} near {middle:.5g} | {status}' + (' | volume-mean-free' if mean_free else ''))
    figure.savefig(directory / f'{name}.png', dpi=110)
    if has_truth:
        metrics = comparison_metrics(predicted_dp, previous_dp, truth_dp, volumes, prediction_mode)
        if not write_metrics:
            return metrics
        path = directory / 'prediction_metrics.csv'
        exists = path.exists() and append_metrics
        with path.open('a' if exists else 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['sample', 'gauge'] + list(metrics))
            if not exists:
                writer.writeheader()
            writer.writerow(dict(sample=name, gauge='volume_mean_free' if mean_free else 'provided', **metrics))
        return metrics
    return None


def plot_prediction_statistics(predicted_dp, previous_dp, truth_dp, volumes, directory, name,
                               mesh_label='whole coarse graph'):
    weights = np.asarray(volumes).reshape(-1)
    weights = weights / weights.sum()
    fields = [np.asarray(field).reshape(-1).copy() for field in (predicted_dp, previous_dp, truth_dp)]
    for field in fields:
        field -= np.sum(field * weights)
    predicted, previous, truth = fields
    selected = np.random.default_rng(42).choice(len(truth), min(len(truth), 10000), replace=False)
    figure, axes = _figure(2, 3)
    for row, label in enumerate(('dp', 'ddp')):
        target = truth if label == 'dp' else truth - previous
        prediction = predicted if label == 'dp' else predicted - previous
        baseline = previous if label == 'dp' else np.zeros_like(previous)
        previous_only = np.zeros_like(previous) if label == 'dp' else -previous
        for column, (values, title) in enumerate(((prediction, 'SM'), (baseline, 'dp = dpPrev'))):
            axis = axes[row, column]
            low = float(min(target.min(), values.min()))
            high = float(max(target.max(), values.max()))
            if low == high:
                low, high = low - 1e-12, high + 1e-12
            axis.scatter(target[selected], values[selected], s=2, alpha=.3, linewidths=0)
            axis.plot([low, high], [low, high], color='black', linewidth=.8)
            axis.set(title=f'{label}: {title} parity', xlabel='CFD target', ylabel='Prediction', aspect='equal')
        axis = axes[row, 2]
        limit = max(float(np.max(np.abs(np.concatenate((prediction - target, baseline - target, previous_only - target))))), 1e-12)
        bins = np.linspace(-limit, limit, 61)
        for values, title in ((prediction, 'SM'), (baseline, 'dp = dpPrev'), (previous_only, 'p_prev')):
            axis.hist(values - target, bins=bins, weights=weights, histtype='step', label=title)
        axis.set(title=f'{label}: error distribution', xlabel='Signed error (physical units)', ylabel='Cell-volume fraction')
        axis.set_yscale('log')
        axis.legend()
    figure.suptitle(f'{name} | {mesh_label} | volume-mean-free')
    figure.savefig(Path(directory) / f'{name}_statistics.png', dpi=120)


def plot_resolution_error_distribution(fields, directory, name):
    figure, axes = _figure(1, 2)
    pressure_limit = max(max(float(np.max(np.abs(field['pressure_error']))) for field in fields), 1e-12)
    velocity_limit = max((float(np.max(np.linalg.norm(field['velocity_error'], axis=1)))
        for field in fields if field.get('velocity_error') is not None), default=1e-12)
    velocity_limit = max(velocity_limit, 1e-12)
    for field in fields:
        weights = np.asarray(field['volumes'], dtype=float)
        weights = weights / weights.sum()
        error = field['pressure_error']
        rmse = float(np.sqrt(np.sum(weights * error ** 2)))
        axes[0, 0].hist(error, bins=np.linspace(-pressure_limit, pressure_limit, 81), weights=weights,
                        histtype='step', label=f"{field['label']} (RMSE {rmse:.4g})")
        if field.get('velocity_error') is not None:
            error = np.linalg.norm(field['velocity_error'], axis=1)
            rmse = float(np.sqrt(np.sum(weights * error ** 2)))
            axes[0, 1].hist(error, bins=np.linspace(0., velocity_limit, 81), weights=weights,
                            histtype='step', label=f"{field['label']} (RMSE {rmse:.4g})")
    axes[0, 0].set(title='Pressure increment error (volume-mean-free)', xlabel='Signed error', ylabel='Volume fraction')
    axes[0, 1].set(title='Total velocity increment error', xlabel='Vector error magnitude', ylabel='Volume fraction')
    for axis in axes.flat:
        if axis.get_legend_handles_labels()[0]:
            axis.set_yscale('log')
            axis.legend(loc='upper center', bbox_to_anchor=(.5, -.28), fontsize=8)
    figure.set_size_inches(12, 5)
    figure.set_layout_engine(None)
    figure.subplots_adjust(left=.08, right=.98, top=.85, bottom=.34, wspace=.3)
    figure.suptitle(f'{name} | graph versus full CFD mesh | physical units, all cells')
    figure.savefig(Path(directory) / f'{name}_error_distribution.png', dpi=120)


def obstacle_distance_pressure_metrics(sample, pressure_error, baseline_error, distance, centres, volumes):
    pressure_error = np.asarray(pressure_error, dtype=float).reshape(-1)
    baseline_error = np.asarray(baseline_error, dtype=float).reshape(-1)
    distance = np.asarray(distance, dtype=float).reshape(-1)
    centres = np.asarray(centres, dtype=float)
    volumes = np.asarray(volumes, dtype=float).reshape(-1)
    count = len(volumes)
    if (pressure_error.shape != (count,) or baseline_error.shape != (count,)
            or distance.shape != (count,) or centres.shape != (count, 3)
            or not np.isfinite(pressure_error).all() or not np.isfinite(baseline_error).all()
            or not np.isfinite(distance).all() or np.any(distance < 0)
            or not np.isfinite(volumes).all() or np.any(volumes <= 0)):
        raise ValueError('Obstacle-error arrays must be finite and match the native mesh.')
    distance_in_cell_widths = distance / np.cbrt(volumes)
    edges = (0., 1., 2., 4., 8., np.inf)
    labels = ('<1', '1-2', '2-4', '4-8', '>=8')
    rows = []
    total_volume = float(volumes.sum())
    for lower, upper, label in zip(edges[:-1], edges[1:], labels):
        mask = ((distance_in_cell_widths >= lower) & (distance_in_cell_widths < upper)
                if np.isfinite(upper) else distance_in_cell_widths >= lower)
        selected_volume = float(volumes[mask].sum())
        if not mask.any():
            rows.append(dict(sample=sample, distance_band=label, cells=0, volume_fraction=0.,
                             mean_distance_m=None, pressure_rmse=None, pressure_mae=None,
                             baseline_rmse=None, improvement_percent=None))
            continue
        weights = volumes[mask] / selected_volume
        rmse = float(np.sqrt(np.sum(weights * pressure_error[mask] ** 2)))
        baseline_rmse = float(np.sqrt(np.sum(weights * baseline_error[mask] ** 2)))
        rows.append(dict(sample=sample, distance_band=label, cells=int(mask.sum()),
                         volume_fraction=selected_volume / total_volume,
                         mean_distance_m=float(np.average(distance[mask], weights=volumes[mask])),
                         pressure_rmse=rmse,
                         pressure_mae=float(np.sum(weights * np.abs(pressure_error[mask]))),
                         baseline_rmse=baseline_rmse,
                         improvement_percent=(100. * (1. - rmse / baseline_rmse)
                                              if baseline_rmse > 0 else None)))
    return rows


def plot_obstacle_pressure_error_map(centres, pressure_error, obstacle_points, directory, name,
                                     max_points=30000):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    centres = np.asarray(centres, dtype=float)
    error = np.asarray(pressure_error, dtype=float).reshape(-1)
    obstacle_points = np.asarray(obstacle_points, dtype=float).reshape(-1, 3)
    if centres.shape != (len(error), 3) or not np.isfinite(centres).all() or not np.isfinite(error).all():
        raise ValueError('Obstacle pressure-error map requires finite native-cell data.')
    z_levels = np.unique(centres[:, 2])
    z_middle = .5 * (z_levels[0] + z_levels[-1])
    z_value = z_levels[np.argmin(np.abs(z_levels - z_middle))]
    tolerance = max(1e-12, .05 * float(np.median(np.diff(z_levels)))) if len(z_levels) > 1 else 1e-12
    selected = np.flatnonzero(np.abs(centres[:, 2] - z_value) <= tolerance)
    if len(selected) > max_points:
        selected = np.random.default_rng(42).choice(selected, max_points, replace=False)
    obstacle_slice = obstacle_points[np.abs(obstacle_points[:, 2] - z_value) <= tolerance]
    limit = max(float(np.percentile(np.abs(error[selected]), 99.5)), 1e-12)
    figure, axes = _figure(1, 2)
    for axis, values, title, cmap, vmin, vmax in (
        (axes[0, 0], error[selected], 'Signed pressure error', 'RdBu_r', -limit, limit),
        (axes[0, 1], np.abs(error[selected]), 'Absolute pressure error', 'magma', 0., limit),
    ):
        image = axis.scatter(centres[selected, 0], centres[selected, 1], c=values, s=4,
                             cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0, rasterized=True)
        if len(obstacle_slice):
            axis.scatter(obstacle_slice[:, 0], obstacle_slice[:, 1], s=7, c='cyan',
                         edgecolors='black', linewidths=.25, label='Obstacle boundary')
            axis.legend(loc='best', fontsize=7)
        axis.set(title=title, xlabel='x', ylabel='y', aspect='equal')
        figure.colorbar(image, ax=axis, shrink=.8, label='Pressure error')
    figure.suptitle(f'{name} | native-cell pressure error near z={z_value:.5g}; obstacle overlaid')
    figure.savefig(directory / f'{name}_obstacle_pressure_error.png', dpi=130)


def plot_obstacle_distance_pressure_summary(records, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    labels = ('<1', '1-2', '2-4', '4-8', '>=8')
    grouped = {label: [row for row in records
                       if row['distance_band'] == label and row['pressure_rmse'] is not None]
               for label in labels}
    available = [label for label in labels if grouped[label]]
    if not available:
        return
    positions = np.arange(len(available))

    def mean_and_std(field):
        values = [np.asarray([row[field] for row in grouped[label] if row[field] is not None], dtype=float)
                  for label in available]
        return (np.asarray([np.mean(value) if len(value) else np.nan for value in values]),
            np.asarray([np.std(value) if len(value) else np.nan for value in values]))

    pressure_mean, pressure_std = mean_and_std('pressure_rmse')
    baseline_mean, baseline_std = mean_and_std('baseline_rmse')
    improvement_mean, improvement_std = mean_and_std('improvement_percent')
    figure, axes = _figure(1, 2)
    axes[0, 0].errorbar(positions, pressure_mean, yerr=pressure_std, marker='o', capsize=3, label='SM')
    axes[0, 0].errorbar(positions, baseline_mean, yerr=baseline_std, marker='s', capsize=3,
                        label='dpPrev baseline')
    axes[0, 0].set(title='Pressure RMSE by obstacle distance', ylabel='Volume-weighted RMSE',
                   xlabel='Distance / local cell width', xticks=positions, xticklabels=available)
    axes[0, 0].legend()
    valid_improvement = np.isfinite(improvement_mean)
    if valid_improvement.any():
        axes[0, 1].errorbar(positions[valid_improvement], improvement_mean[valid_improvement],
                            yerr=improvement_std[valid_improvement], marker='o', capsize=3)
    axes[0, 1].axhline(0., color='black', linewidth=.8)
    axes[0, 1].set(title='Improvement over dpPrev', ylabel='RMSE improvement (%)',
                   xlabel='Distance / local cell width', xticks=positions, xticklabels=available)
    for axis in axes.ravel():
        axis.grid(alpha=.2)
    figure.suptitle('Held-out pressure error versus obstacle proximity; error bars show timestep variation')
    figure.savefig(directory / 'obstacle_pressure_error_by_distance.png', dpi=140)


def plot_shifter_components(centres, outputs, gradient, target_scale, pressure_scale, directory, name):
    outputs = np.asarray(outputs)
    shifted = -np.sum(outputs[:, :3] * np.asarray(gradient), axis=1) * pressure_scale
    source = outputs[:, 3] * target_scale * pressure_scale
    fields = [outputs[:, 0], outputs[:, 1], outputs[:, 2], shifted, source, shifted + source]
    titles = ['Shift x (model units)', 'Shift y (model units)', 'Shift z (model units)',
              'Gradient-shift contribution', 'Source contribution', 'Combined (before mean removal)']
    middle = .5 * (centres[:, 2].min() + centres[:, 2].max())
    selected = np.argsort(np.abs(centres[:, 2] - middle))[:min(10000, len(centres))]
    figure, axes = _figure(2, 3)
    for axis, field, title in zip(axes.ravel(), fields, titles):
        limit = max(float(np.max(np.abs(field[selected]))), 1e-12)
        image = axis.scatter(centres[selected, 0], centres[selected, 1], c=field[selected], s=3,
                             cmap='RdBu_r', vmin=-limit, vmax=limit, linewidths=0)
        axis.set(title=title, xlabel='x', ylabel='y', aspect='equal')
        figure.colorbar(image, ax=axis, shrink=.75)
    figure.suptitle(f'{name} | shifter decomposition, z near {middle:.5g} | pressure contributions in physical units')
    figure.savefig(Path(directory) / f'{name}_shifter.png', dpi=120)


def plot_velocity_comparison(centres, prediction, truth, directory, name, previous=None,
                            residual_prediction=None, residual_truth=None, volumes=None,
                            start_velocity=None, mesh_label='whole coarse graph'):
    centres = np.asarray(centres)
    selected = np.argsort(np.abs(centres[:, 2] - np.median(centres[:, 2])))[:min(10000, len(centres))]
    prediction = np.asarray(prediction)
    truth = np.asarray(truth)
    error = prediction - truth
    weights = np.ones(len(centres), dtype=float) if volumes is None else np.asarray(volumes, dtype=float).reshape(-1)
    if weights.shape != (len(centres),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Velocity plot volumes must be finite, positive, and match the cell count.')
    weights /= weights.sum()

    def plot_statistics(target, predicted, baselines, filename, title):
        statistics_selected = np.random.default_rng(42).choice(len(centres), min(10000, len(centres)), replace=False)
        figure, axes = _figure(3, 2 + len(baselines))
        for component in range(3):
            baseline_fields = [field for _, field in baselines]
            low = float(min([target[:, component].min(), predicted[:, component].min()]
                            + [field[:, component].min() for field in baseline_fields]))
            high = float(max([target[:, component].max(), predicted[:, component].max()]
                             + [field[:, component].max() for field in baseline_fields]))
            if low == high:
                low, high = low - 1e-12, high + 1e-12
            axis = axes[component, 0]
            axis.scatter(target[statistics_selected, component], predicted[statistics_selected, component], s=2, alpha=.3, linewidths=0)
            axis.plot([low, high], [low, high], color='black', linewidth=.8)
            axis.set(title=f'{"xyz"[component]}: SM parity', xlabel='CFD target', ylabel='Prediction', aspect='equal')

            for baseline_index, (baseline_label, baseline) in enumerate(baselines, start=1):
                axis = axes[component, baseline_index]
                axis.scatter(target[statistics_selected, component], baseline[statistics_selected, component], s=2, alpha=.3, linewidths=0)
                axis.plot([low, high], [low, high], color='black', linewidth=.8)
                axis.set(title=f'{"xyz"[component]}: {baseline_label} parity', xlabel='CFD target',
                         ylabel='Prediction', aspect='equal')

            axis = axes[component, -1]
            prediction_error = predicted[:, component] - target[:, component]
            baseline_errors = [(label, baseline[:, component] - target[:, component])
                               for label, baseline in baselines]
            combined_errors = np.concatenate([prediction_error] + [error for _, error in baseline_errors])
            limit = max(float(np.max(np.abs(combined_errors))), 1e-12)
            bins = np.linspace(-limit, limit, 61)
            axis.hist(prediction_error, bins=bins, weights=weights, histtype='step', label='SM')
            for baseline_label, baseline_error in baseline_errors:
                axis.hist(baseline_error, bins=bins, weights=weights,
                          histtype='step', label=baseline_label)
            axis.set(title=f'{"xyz"[component]}: error distribution', xlabel='Signed error (physical units)',
                     ylabel='Cell-volume fraction')
            axis.set_yscale('log')
            axis.legend()
        figure.suptitle(title)
        figure.savefig(Path(directory) / filename, dpi=120)

    fields = [('CFD total dU', truth), ('SM total dU', prediction)]
    if previous is not None:
        fields.append(('dUPrev baseline', np.asarray(previous)))
    fields.append(('SM - CFD', error))
    figure, axes = _figure(3, len(fields))
    for component in range(3):
        field_limit = max(float(np.percentile(np.abs(np.concatenate(
            [field[selected, component] for _, field in fields[:-1]])), 99.)), 1e-12)
        error_limit = max(float(np.percentile(np.abs(error[selected, component]), 99.)), 1e-12)
        for column, (title, field) in enumerate(fields):
            axis = axes[component, column]
            image = axis.scatter(centres[selected, 0], centres[selected, 1], c=field[selected, component],
                                 s=3, cmap='RdBu_r',
                                 vmin=-(error_limit if title == 'SM - CFD' else field_limit),
                                 vmax=error_limit if title == 'SM - CFD' else field_limit,
                                 linewidths=0)
            axis.set(title=f'{title}: {"xyz"[component]}', xlabel='x', ylabel='y', aspect='equal')
            figure.colorbar(image, ax=axis, shrink=.75)
    figure.suptitle(f'{name} | total timestep velocity increment, physical velocity units; central z slice')
    figure.savefig(Path(directory) / f'{name}_velocity.png', dpi=120)
    if start_velocity is not None:
        start_velocity = np.asarray(start_velocity)
        state_prediction = start_velocity + prediction
        state_truth = start_velocity + truth
        state_baselines = [('U_prev', start_velocity)]
        if previous is not None:
            state_baselines.append(('U_prev + dUPrev', start_velocity + np.asarray(previous)))
        plot_statistics(state_truth, state_prediction, state_baselines,
                        f'{name}_velocity_statistics.png',
                        f'{name} | final U, {mesh_label}, volume-weighted diagnostics')
    elif previous is not None:
        plot_statistics(truth, prediction, [('dUPrev', np.asarray(previous))],
                        f'{name}_velocity_statistics.png',
                        f'{name} | total dU, {mesh_label}, volume-weighted diagnostics')

    if residual_prediction is not None and residual_truth is not None:
        residual_prediction = np.asarray(residual_prediction)
        residual_truth = np.asarray(residual_truth)
        residual_error = residual_prediction - residual_truth
        residual_fields = (('CFD ddU target', residual_truth),
                           ('SM ddU prediction', residual_prediction),
                           ('SM - CFD ddU', residual_error))
        figure, axes = _figure(3, len(residual_fields))
        for component in range(3):
            field_limit = max(float(np.percentile(np.abs(np.concatenate(
                [field[selected, component] for _, field in residual_fields[:-1]])), 99.)), 1e-12)
            error_limit = max(float(np.percentile(np.abs(residual_error[selected, component]), 99.)), 1e-12)
            for column, (title, field) in enumerate(residual_fields):
                axis = axes[component, column]
                limit = error_limit if title == 'SM - CFD ddU' else field_limit
                image = axis.scatter(centres[selected, 0], centres[selected, 1], c=field[selected, component],
                                     s=3, cmap='RdBu_r', vmin=-limit, vmax=limit, linewidths=0)
                axis.set(title=f'{title}: {"xyz"[component]}', xlabel='x', ylabel='y', aspect='equal')
                figure.colorbar(image, ax=axis, shrink=.75)
        figure.suptitle(f'{name} | residual velocity increment ddU, physical velocity units; central z slice')
        figure.savefig(Path(directory) / f'{name}_velocity_residual.png', dpi=120)
        plot_statistics(residual_truth, residual_prediction,
                [('zero ddU', np.zeros_like(residual_truth))],
                f'{name}_velocity_residual_statistics.png',
                f'{name} | residual ddU, {mesh_label}, volume-weighted diagnostics')


def plot_validation_history(records, directory):
    figure, axes = _figure(1, 3)
    indices = np.arange(1, len(records) + 1)
    for key, title in (('sm_dp', 'SM'), ('p_prev', 'p_prev'), ('pxtrap', 'pExtrap'),
                       ('extrapolation_dp', 'p_prev + dpPrev')):
        axes[0, 0].plot(indices, [record[f'{key}_rmse'] for record in records], marker='.', label=title)
        axes[0, 1].plot(indices, [record[f'{key}_mae'] for record in records], marker='.', label=title)
    for key, title in (('p_prev', 'p_prev'), ('pxtrap', 'pExtrap')):
        axes[0, 2].plot(indices, [record[f'improvement_vs_{key}_percent'] for record in records], marker='.', label=title)
    axes[0, 2].axhline(0, color='black', linewidth=.8)
    for axis, title, ylabel in zip(axes.ravel(), ('Held-out RMSE', 'Held-out MAE', 'SM improvement'),
                                   ('Pressure units', 'Pressure units', 'Percent; positive is better')):
        axis.set(title=title, xlabel='Held-out timestep index', ylabel=ylabel)
        axis.legend()
    figure.savefig(Path(directory) / 'validation_history.png', dpi=120)