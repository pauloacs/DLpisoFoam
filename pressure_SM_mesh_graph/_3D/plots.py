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


def plot_prediction_statistics(predicted_dp, previous_dp, truth_dp, volumes, directory, name):
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
    figure.suptitle(f'{name} | whole coarse graph | volume-mean-free')
    figure.savefig(Path(directory) / f'{name}_statistics.png', dpi=120)


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


def plot_velocity_comparison(centres, prediction, truth, directory, name):
    selected = np.argsort(np.abs(centres[:, 2] - np.median(centres[:, 2])))[:min(10000, len(centres))]
    figure, axes = _figure(3, 3)
    for component in range(3):
        limit = max(float(np.max(np.abs(np.concatenate((prediction[:, component], truth[:, component]))))), 1e-12)
        for column, (field, title) in enumerate(((truth, 'CFD dU'), (prediction, 'SM dU'), (prediction - truth, 'SM - CFD'))):
            axis = axes[component, column]
            image = axis.scatter(centres[selected, 0], centres[selected, 1], c=field[selected, component],
                                 s=3, cmap='RdBu_r', vmin=-limit, vmax=limit, linewidths=0)
            axis.set(title=f'{title}: {"xyz"[component]}', xlabel='x', ylabel='y', aspect='equal')
            figure.colorbar(image, ax=axis, shrink=.75)
    figure.suptitle(f'{name} | total timestep velocity increment, physical velocity units')
    figure.savefig(Path(directory) / f'{name}_velocity.png', dpi=120)


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