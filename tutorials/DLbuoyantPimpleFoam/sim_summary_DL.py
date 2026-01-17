import re
import csv
import os 
import numpy as np

def parse_logs(log_content):
    # Regular expression to find each Time block and capture the necessary data for DLbuoyantPimpleFoam
    time_block_pattern = re.compile(
        r'Time = (?P<time>\d+\.\d+)\n'
        r'.*?diagonal:\s+Solving for rho, Initial residual = .*?, Final residual = .*?, No Iterations (?P<rho_iterations>\d+)\n'
        r'smoothSolver:\s+Solving for Ux, Initial residual = (?P<ux_initial_residual>\d+\.?\d*e?-?\d*), Final residual = .*?, No Iterations (?P<ux_iterations>\d+)\n'
        r'smoothSolver:\s+Solving for Uy, Initial residual = (?P<uy_initial_residual>\d+\.?\d*e?-?\d*), Final residual = .*?, No Iterations (?P<uy_iterations>\d+)\n'
        r'smoothSolver:\s+Solving for Uz, Initial residual = (?P<uz_initial_residual>\d+\.?\d*e?-?\d*), Final residual = .*?, No Iterations (?P<uz_iterations>\d+)\n'
        r'DILUPBiCGStab:\s+Solving for h, Initial residual = (?P<h_initial_residual>\d+\.?\d*e?-?\d*), Final residual = .*?, No Iterations (?P<h_iterations>\d+)\n'
        r'>>> Setting arguments <<<\n'
        r'>>> Calling python function <<<\n'
        r'.*?'
        r'>>>  delta_p_rgh filled <<<\n'
        r'DL pressure prediction & data transport: (?P<dl_time>\d+\.\d+) ms\n'
        r'GAMG:\s+Solving for p_rgh, Initial residual = (?P<p_iter1_1_initial_residual>\d+\.?\d*e?-?\d*), Final residual = .*?, No Iterations (?P<p_iter1_1_iterations>\d+)\n'
        r'GAMG:\s+Solving for p_rgh, Initial residual = (?P<p_iter1_2_initial_residual>\d+\.?\d*e?-?\d*), Final residual = .*?, No Iterations (?P<p_iter1_2_iterations>\d+)\n'
        r'diagonal:\s+Solving for rho, Initial residual = .*?, Final residual = .*?, No Iterations (?P<rho2_iterations>\d+)\n'
        r'time step continuity errors : sum local = .*?, global = .*?, cumulative = .*?\n'
        r'GAMG:\s+Solving for p_rgh, Initial residual = (?P<p_iter2_1_initial_residual>\d+\.?\d*e?-?\d*), Final residual = .*?, No Iterations (?P<p_iter2_1_iterations>\d+)\n'
        r'GAMG:\s+Solving for p_rgh, Initial residual = (?P<p_iter2_2_initial_residual>\d+\.?\d*e?-?\d*), Final residual = .*?, No Iterations (?P<p_iter2_2_iterations>\d+)\n'
        r'diagonal:\s+Solving for rho, Initial residual = .*?, Final residual = .*?, No Iterations (?P<rho3_iterations>\d+)\n'
        r'time step continuity errors : sum local = .*?, global = .*?, cumulative = .*?\n'
        r'smoothSolver:\s+Solving for omega, Initial residual = (?P<omega_initial_residual>\d+\.?\d*e?-?\d*), Final residual = .*?, No Iterations (?P<omega_iterations>\d+)\n'
        r'smoothSolver:\s+Solving for k, Initial residual = (?P<k_initial_residual>\d+\.?\d*e?-?\d*), Final residual = .*?, No Iterations (?P<k_iterations>\d+)\n'
        r'ExecutionTime = (?P<execution_time>\d+\.\d+) s',
        re.DOTALL
    )

    # Extract all matches
    matches = time_block_pattern.findall(log_content)

    return matches



def save_initial_residuals(matches, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the header
        csv_writer.writerow([
            'Time', 
            'Ux_iterations', 'Uy_iterations', 'Uz_iterations',
            'h_iterations',
            'DL_time_ms',
            'p_iter1_1_initial_residual', 'p_iter1_1_iterations', 
            'p_iter1_2_iterations', 
            'p_iter2_1_iterations', 
            'p_iter2_2_iterations',
            'omega_iterations',
            'k_iterations',
            'execution_time_s'
        ])
        # Write the data
        for match in matches:
            time, rho_iterations, \
            ux_initial_residual, ux_iterations, \
            uy_initial_residual, uy_iterations, \
            uz_initial_residual, uz_iterations, \
            h_initial_residual, h_iterations, \
            dl_time, \
            p_iter1_1_initial_residual, p_iter1_1_iterations, \
            p_iter1_2_initial_residual, p_iter1_2_iterations, \
            rho2_iterations, \
            p_iter2_1_initial_residual, p_iter2_1_iterations, \
            p_iter2_2_initial_residual, p_iter2_2_iterations, \
            rho3_iterations, \
            omega_initial_residual, omega_iterations, \
            k_initial_residual, k_iterations, \
            execution_time = match

            csv_writer.writerow([
                time, 
                ux_iterations, uy_iterations, uz_iterations,
                h_iterations,
                dl_time,
                p_iter1_1_initial_residual, p_iter1_1_iterations, 
                p_iter1_2_iterations, 
                p_iter2_1_iterations, 
                p_iter2_2_iterations,
                omega_iterations,
                k_iterations,
                execution_time
            ])

def calculate_averages(matches):
    """Calculate average values for the simulation"""
    if not matches:
        return None
    
    # Extract numeric data
    ux_iters = [int(m[3]) for m in matches]
    uy_iters = [int(m[5]) for m in matches]
    uz_iters = [int(m[7]) for m in matches]
    h_iters = [int(m[9]) for m in matches]
    dl_times = [float(m[10]) for m in matches]
    p_iter1_1_iters = [int(m[12]) for m in matches]
    p_iter1_2_iters = [int(m[14]) for m in matches]
    p_iter2_1_iters = [int(m[17]) for m in matches]
    p_iter2_2_iters = [int(m[19]) for m in matches]
    omega_iters = [int(m[22]) for m in matches]
    k_iters = [int(m[24]) for m in matches]
    execution_times = [float(m[25]) for m in matches]
    
    # Calculate time per iteration (difference between consecutive execution times)
    time_per_iter = [execution_times[i] - execution_times[i-1] if i > 0 else execution_times[0] 
                     for i in range(len(execution_times))]
    
    # Calculate total pressure iterations per timestep
    total_p_iters = [p_iter1_1_iters[i] + p_iter1_2_iters[i] + p_iter2_1_iters[i] + p_iter2_2_iters[i] 
                     for i in range(len(matches))]
    
    averages = {
        'num_timesteps': len(matches),
        'avg_ux_iterations': np.mean(ux_iters),
        'avg_uy_iterations': np.mean(uy_iters),
        'avg_uz_iterations': np.mean(uz_iters),
        'avg_h_iterations': np.mean(h_iters),
        'avg_dl_time_ms': np.mean(dl_times),
        'avg_p_iter1_1_iterations': np.mean(p_iter1_1_iters),
        'avg_p_iter1_2_iterations': np.mean(p_iter1_2_iters),
        'avg_p_iter2_1_iterations': np.mean(p_iter2_1_iters),
        'avg_p_iter2_2_iterations': np.mean(p_iter2_2_iters),
        'avg_total_p_iterations': np.mean(total_p_iters),
        'avg_omega_iterations': np.mean(omega_iters),
        'avg_k_iterations': np.mean(k_iters),
        'avg_time_per_iteration': np.mean(time_per_iter),
        'total_execution_time': execution_times[-1] if execution_times else 0,
    }
    
    return averages

# Read the log content from a file
with open('./DL.log', 'r') as file:
    log_content = file.read()

import pdb; pdb.set_trace()
# Parse the logs to get the time and initial residuals
matches = parse_logs(log_content)

# Folder name (assuming it's the current directory)
folder_name = os.path.basename(os.getcwd())

# Save the initial residuals to a CSV file
output_file = f'DLbuoyantPimpleFoam_sim{folder_name}_summary.csv'
save_initial_residuals(matches, output_file)

# Calculate and print averages
averages = calculate_averages(matches)
if averages:
    print(f'\n=== Simulation Summary (DLbuoyantPimpleFoam) ===')
    print(f'Parsed {averages["num_timesteps"]} timesteps')
    print(f'\nAverage Iterations:')
    print(f'  Ux: {averages["avg_ux_iterations"]:.2f}')
    print(f'  Uy: {averages["avg_uy_iterations"]:.2f}')
    print(f'  Uz: {averages["avg_uz_iterations"]:.2f}')
    print(f'  h:  {averages["avg_h_iterations"]:.2f}')
    print(f'  omega: {averages["avg_omega_iterations"]:.2f}')
    print(f'  k:     {averages["avg_k_iterations"]:.2f}')
    print(f'\nAverage Pressure Iterations:')
    print(f'  Iter1_1: {averages["avg_p_iter1_1_iterations"]:.2f}')
    print(f'  Iter1_2: {averages["avg_p_iter1_2_iterations"]:.2f}')
    print(f'  Iter2_1: {averages["avg_p_iter2_1_iterations"]:.2f}')
    print(f'  Iter2_2: {averages["avg_p_iter2_2_iterations"]:.2f}')
    print(f'  Total:   {averages["avg_total_p_iterations"]:.2f}')
    print(f'\nAverage DL prediction time: {averages["avg_dl_time_ms"]:.2f} ms')
    print(f'\nExecution Time:')
    print(f'  Average per iteration: {averages["avg_time_per_iteration"]:.2f} s')
    print(f'  Total execution time:  {averages["total_execution_time"]:.2f} s')
    print(f'\nResults saved to: {output_file}')
else:
    print('No timesteps were parsed successfully.')
