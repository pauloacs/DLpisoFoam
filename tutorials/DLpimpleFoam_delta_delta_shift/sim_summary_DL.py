import re
import csv
import os
import sys
import numpy as np

# Generic float regex: supports decimal and scientific notation
FLOAT = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'


def parse_logs(log_content, start_time=None, end_time=None):
    """
    Parse OpenFOAM/DLpisoFoam logs.

    Supports:
    - Time = ...
    - bothTime = ...
    - deltaT = ...
    - optional DL pressure prediction section
    - 2 or 3 pressure correctors

    Filters timesteps to keep only:
        start_time <= time <= end_time
    if start_time / end_time are provided.
    """

    # Split log into timestep blocks
    block_start_pattern = re.compile(
        rf'(?=^(?:Time|bothTime)\s*=\s*{FLOAT}\s*$)',
        re.MULTILINE
    )

    starts = [m.start() for m in block_start_pattern.finditer(log_content)]
    if not starts:
        return []

    starts.append(len(log_content))
    blocks = [log_content[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]

    parsed = []

    for block in blocks:
        entry = {}

        # --- Time / bothTime
        m = re.search(
            rf'^(?:Time|bothTime)\s*=\s*(?P<time>{FLOAT})\s*$',
            block,
            re.MULTILINE
        )
        if not m:
            continue

        time_value = float(m.group('time'))

        # Apply time filtering
        if start_time is not None and time_value < start_time:
            continue
        if end_time is not None and time_value > end_time:
            continue

        entry['time'] = time_value

        # --- Courant number
        m = re.search(
            rf'Courant Number mean:\s*(?P<mean>{FLOAT})\s+max:\s*(?P<max>{FLOAT})',
            block
        )
        if m:
            entry['courant_mean'] = float(m.group('mean'))
            entry['courant_max'] = float(m.group('max'))
        else:
            entry['courant_mean'] = np.nan
            entry['courant_max'] = np.nan

        # --- deltaT
        m = re.search(rf'deltaT\s*=\s*(?P<deltaT>{FLOAT})', block)
        entry['deltaT'] = float(m.group('deltaT')) if m else np.nan

        # --- Velocity solves (first occurrence of Ux/Uy/Uz)
        for field in ['Ux', 'Uy', 'Uz']:
            m = re.search(
                rf'smoothSolver:\s+Solving for {field}, Initial residual = (?P<init>{FLOAT}), '
                rf'Final residual = (?P<final>{FLOAT}), No Iterations (?P<it>\d+)',
                block
            )
            if m:
                entry[f'{field}_initial_residual'] = float(m.group('init'))
                entry[f'{field}_iterations'] = int(m.group('it'))
            else:
                entry[f'{field}_initial_residual'] = np.nan
                entry[f'{field}_iterations'] = np.nan

        # --- Optional DL prediction timing
        m = re.search(
            rf'DL pressure prediction(?:\s*&\s*data transport)?:\s*(?P<dl_time>{FLOAT})\s*ms',
            block
        )
        entry['dl_time_ms'] = float(m.group('dl_time')) if m else np.nan

        # --- Pressure solves
        p_matches = re.findall(
            rf'GAMG:\s+Solving for p, Initial residual = ({FLOAT}), '
            rf'Final residual = ({FLOAT}), No Iterations (\d+)',
            block
        )

        max_correctors = 3
        for corr in range(1, max_correctors + 1):
            for sub in [1, 2]:
                idx = (corr - 1) * 2 + (sub - 1)
                key_res = f'p_iter{corr}_{sub}_initial_residual'
                key_it = f'p_iter{corr}_{sub}_iterations'

                if idx < len(p_matches):
                    init_res, final_res, iters = p_matches[idx]
                    entry[key_res] = float(init_res)
                    entry[key_it] = int(iters)
                else:
                    entry[key_res] = np.nan
                    entry[key_it] = np.nan

        entry['num_pressure_solves'] = len(p_matches)
        entry['num_pressure_correctors'] = len(p_matches) // 2

        # --- omega
        m = re.search(
            rf'smoothSolver:\s+Solving for omega, Initial residual = (?P<init>{FLOAT}), '
            rf'Final residual = (?P<final>{FLOAT}), No Iterations (?P<it>\d+)',
            block
        )
        if m:
            entry['omega_initial_residual'] = float(m.group('init'))
            entry['omega_iterations'] = int(m.group('it'))
        else:
            entry['omega_initial_residual'] = np.nan
            entry['omega_iterations'] = np.nan

        # --- k
        m = re.search(
            rf'smoothSolver:\s+Solving for k, Initial residual = (?P<init>{FLOAT}), '
            rf'Final residual = (?P<final>{FLOAT}), No Iterations (?P<it>\d+)',
            block
        )
        if m:
            entry['k_initial_residual'] = float(m.group('init'))
            entry['k_iterations'] = int(m.group('it'))
        else:
            entry['k_initial_residual'] = np.nan
            entry['k_iterations'] = np.nan

        # --- ExecutionTime
        m = re.search(rf'ExecutionTime\s*=\s*(?P<execution_time>{FLOAT})\s*s', block)
        entry['execution_time_s'] = float(m.group('execution_time')) if m else np.nan

        parsed.append(entry)

    return parsed


def save_initial_residuals(matches, output_file):
    headers = [
        'Time',
        'deltaT',
        'Courant_mean',
        'Courant_max',
        'Ux_iterations', 'Uy_iterations', 'Uz_iterations',
        'DL_time_ms',

        'p_iter1_1_initial_residual', 'p_iter1_1_iterations',
        'p_iter1_2_initial_residual', 'p_iter1_2_iterations',

        'p_iter2_1_initial_residual', 'p_iter2_1_iterations',
        'p_iter2_2_initial_residual', 'p_iter2_2_iterations',

        'p_iter3_1_initial_residual', 'p_iter3_1_iterations',
        'p_iter3_2_initial_residual', 'p_iter3_2_iterations',

        'num_pressure_correctors',
        'omega_iterations',
        'k_iterations',
        'execution_time_s'
    ]

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for m in matches:
            writer.writerow([
                m.get('time', np.nan),
                m.get('deltaT', np.nan),
                m.get('courant_mean', np.nan),
                m.get('courant_max', np.nan),

                m.get('Ux_iterations', np.nan),
                m.get('Uy_iterations', np.nan),
                m.get('Uz_iterations', np.nan),

                m.get('dl_time_ms', np.nan),

                m.get('p_iter1_1_initial_residual', np.nan),
                m.get('p_iter1_1_iterations', np.nan),
                m.get('p_iter1_2_initial_residual', np.nan),
                m.get('p_iter1_2_iterations', np.nan),

                m.get('p_iter2_1_initial_residual', np.nan),
                m.get('p_iter2_1_iterations', np.nan),
                m.get('p_iter2_2_initial_residual', np.nan),
                m.get('p_iter2_2_iterations', np.nan),

                m.get('p_iter3_1_initial_residual', np.nan),
                m.get('p_iter3_1_iterations', np.nan),
                m.get('p_iter3_2_initial_residual', np.nan),
                m.get('p_iter3_2_iterations', np.nan),

                m.get('num_pressure_correctors', np.nan),
                m.get('omega_iterations', np.nan),
                m.get('k_iterations', np.nan),
                m.get('execution_time_s', np.nan)
            ])


def calculate_averages(matches):
    """Calculate average values for the simulation."""
    if not matches:
        return None

    def valid(vals):
        return [v for v in vals if not (isinstance(v, float) and np.isnan(v))]

    ux_iters = valid([m.get('Ux_iterations', np.nan) for m in matches])
    uy_iters = valid([m.get('Uy_iterations', np.nan) for m in matches])
    uz_iters = valid([m.get('Uz_iterations', np.nan) for m in matches])
    dl_times = valid([m.get('dl_time_ms', np.nan) for m in matches])

    p_iter1_1_iters = valid([m.get('p_iter1_1_iterations', np.nan) for m in matches])
    p_iter1_2_iters = valid([m.get('p_iter1_2_iterations', np.nan) for m in matches])
    p_iter2_1_iters = valid([m.get('p_iter2_1_iterations', np.nan) for m in matches])
    p_iter2_2_iters = valid([m.get('p_iter2_2_iterations', np.nan) for m in matches])
    p_iter3_1_iters = valid([m.get('p_iter3_1_iterations', np.nan) for m in matches])
    p_iter3_2_iters = valid([m.get('p_iter3_2_iterations', np.nan) for m in matches])

    omega_iters = valid([m.get('omega_iterations', np.nan) for m in matches])
    k_iters = valid([m.get('k_iterations', np.nan) for m in matches])

    execution_times = valid([m.get('execution_time_s', np.nan) for m in matches])
    num_pressure_correctors = valid([m.get('num_pressure_correctors', np.nan) for m in matches])

    time_per_iter = []
    for i, t in enumerate(execution_times):
        if i == 0:
            time_per_iter.append(t)
        else:
            time_per_iter.append(t - execution_times[i - 1])

    total_p_iters = []
    for m in matches:
        fields = [
            m.get('p_iter1_1_iterations', np.nan),
            m.get('p_iter1_2_iterations', np.nan),
            m.get('p_iter2_1_iterations', np.nan),
            m.get('p_iter2_2_iterations', np.nan),
            m.get('p_iter3_1_iterations', np.nan),
            m.get('p_iter3_2_iterations', np.nan),
        ]
        total = sum(v for v in fields if not (isinstance(v, float) and np.isnan(v)))
        total_p_iters.append(total)

    averages = {
        'num_timesteps': len(matches),

        'avg_ux_iterations': np.mean(ux_iters) if ux_iters else np.nan,
        'avg_uy_iterations': np.mean(uy_iters) if uy_iters else np.nan,
        'avg_uz_iterations': np.mean(uz_iters) if uz_iters else np.nan,

        'avg_dl_time_ms': np.mean(dl_times) if dl_times else np.nan,

        'avg_p_iter1_1_iterations': np.mean(p_iter1_1_iters) if p_iter1_1_iters else np.nan,
        'avg_p_iter1_2_iterations': np.mean(p_iter1_2_iters) if p_iter1_2_iters else np.nan,
        'avg_p_iter2_1_iterations': np.mean(p_iter2_1_iters) if p_iter2_1_iters else np.nan,
        'avg_p_iter2_2_iterations': np.mean(p_iter2_2_iters) if p_iter2_2_iters else np.nan,
        'avg_p_iter3_1_iterations': np.mean(p_iter3_1_iters) if p_iter3_1_iters else np.nan,
        'avg_p_iter3_2_iterations': np.mean(p_iter3_2_iters) if p_iter3_2_iters else np.nan,

        'avg_total_p_iterations': np.mean(total_p_iters) if total_p_iters else np.nan,
        'avg_num_pressure_correctors': np.mean(num_pressure_correctors) if num_pressure_correctors else np.nan,

        'avg_omega_iterations': np.mean(omega_iters) if omega_iters else np.nan,
        'avg_k_iterations': np.mean(k_iters) if k_iters else np.nan,

        'avg_time_per_iteration': np.mean(time_per_iter) if time_per_iter else np.nan,
        'total_execution_time': execution_times[-1] if execution_times else np.nan,
    }

    return averages


def main():
    if len(sys.argv) != 4:
        print("Usage:")
        print("    python parse_dl_log.py <log_file> <start_time> <last_time>")
        print("")
        print("Example:")
        print("    python parse_dl_log.py DL.log 0.40 1.20")
        sys.exit(1)

    log_file = sys.argv[1]

    try:
        start_time = float(sys.argv[2])
        end_time = float(sys.argv[3])
    except ValueError:
        print("Error: start_time and last_time must be numeric.")
        sys.exit(1)

    if start_time > end_time:
        print("Error: start_time must be <= last_time.")
        sys.exit(1)

    if not os.path.isfile(log_file):
        print(f"Error: file '{log_file}' not found.")
        sys.exit(1)

    with open(log_file, 'r') as file:
        log_content = file.read()

    matches = parse_logs(log_content, start_time, end_time)

    folder_name = os.path.basename(os.getcwd())
    log_base = os.path.splitext(os.path.basename(log_file))[0]
    output_file = f'{log_base}_sim{folder_name}_{start_time}_{end_time}_summary.csv'

    save_initial_residuals(matches, output_file)

    averages = calculate_averages(matches)
    if averages:
        print('\n=== Simulation Summary (DLpisoFoam) ===')
        print(f'Log file: {log_file}')
        print(f'Time range: {start_time} -> {end_time}')
        print(f'Parsed {averages["num_timesteps"]} timesteps')

        print('\nAverage Iterations:')
        print(f'  Ux:    {averages["avg_ux_iterations"]:.2f}')
        print(f'  Uy:    {averages["avg_uy_iterations"]:.2f}')
        print(f'  Uz:    {averages["avg_uz_iterations"]:.2f}')
        print(f'  omega: {averages["avg_omega_iterations"]:.2f}')
        print(f'  k:     {averages["avg_k_iterations"]:.2f}')

        print('\nAverage Pressure Iterations:')
        print(f'  Iter1_1: {averages["avg_p_iter1_1_iterations"]:.2f}')
        print(f'  Iter1_2: {averages["avg_p_iter1_2_iterations"]:.2f}')
        print(f'  Iter2_1: {averages["avg_p_iter2_1_iterations"]:.2f}')
        print(f'  Iter2_2: {averages["avg_p_iter2_2_iterations"]:.2f}')
        print(f'  Iter3_1: {averages["avg_p_iter3_1_iterations"]:.2f}')
        print(f'  Iter3_2: {averages["avg_p_iter3_2_iterations"]:.2f}')
        print(f'  Total:   {averages["avg_total_p_iterations"]:.2f}')
        print(f'  Avg number of correctors: {averages["avg_num_pressure_correctors"]:.2f}')

        if not np.isnan(averages["avg_dl_time_ms"]):
            print(f'\nAverage DL prediction time: {averages["avg_dl_time_ms"]:.2f} ms')
        else:
            print('\nAverage DL prediction time: not available')

        print('\nExecution Time:')
        print(f'  Average per iteration: {averages["avg_time_per_iteration"]:.2f} s')
        print(f'  Total execution time:  {averages["total_execution_time"]:.2f} s')

        print(f'\nResults saved to: {output_file}')
    else:
        print('No timesteps were parsed successfully in the selected time range.')


if __name__ == "__main__":
    main()