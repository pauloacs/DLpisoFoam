import os
from tqdm import *
import subprocess
import random
import numpy as np

scale = 2
grading = 3

# Regarding the channel lengths, it is defined in the gen_blockMeshDict_plate
# Those are domain_length = 0.75, domain_height = 0.1

# Flow condition variables - Re and Ec
U_inlet =np.array([0.1, 0.2, 0.5, 0.75, 1, 2, 5, 7.5, 10, 15])

T_inlet = np.array([15] * 10) + 273.15
Delta_T = np.array([25, 50, 75, 50, 150, 25, 75, 150, 100, 100])
T_obst = T_inlet + Delta_T

# These are plate lengths - L, b
# For reference, H (distance between walls) = 0.1
# Geometry variables
L = np.array([0.0375] * 10)
alpha = [70, 50, 80, 40, 60, 80, 50, 60, 70, 40]
b = [0.0025] * 10
x_cord = [0.2] * 10

Re = (U_inlet * L * 1.269 / 1.716 * 1E5).round(0)
Ec = U_inlet**2 / (1005 * Delta_T)

print(f'Re numbers: {list(Re)}')
print(f'Ec numbers: {list(Ec)}')

delta_T_write = np.round(np.array(L) / np.array(U_inlet), 4)
end_time = delta_T_write * 25

num_runs = len(L)
sim_data_path = 'simulation_data'

# Create the simulation_data directory if it doesn't exist
if not os.path.exists(sim_data_path):
    os.makedirs(sim_data_path)

for i in tqdm(range(num_runs)):
    with open(os.devnull, 'w') as devnull:
        # Remove any previous simulation file
        cmd = f"rm -rf {sim_data_path}/{i}"
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)

        # Copy the OpenFOAM forwardStep directory
        cmd = f"cp -a ./original_snappy/. ./{sim_data_path}/{i}"
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)

        # Remove the blockMeshDict file from system directory
        cmd = f"rm -f ./{sim_data_path}/{i}/system/blockMeshDict"
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)

        # Execute python program to write a blockMeshDict file
        cmd = "python gen_blockMeshDict_plate.py"+ " " + str(x_cord[i]) + " " + str(L[i]) + " " + str(b[i]) + " " + str(alpha[i]) + " " + str(scale) + " " + str(grading)
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)

        # Move the blockMeshDict file to system directory
        cmd = f"mv blockMeshDict ./{sim_data_path}/{i}/system/blockMeshDict_plate"
        subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)

        ## Everything is ready.
        ## Now changing Reynolds (using U_in) and \Delta T (using T_obst and T_in)
        # 
        T_file_path = f"./{sim_data_path}/{i}/0/T"
        with open(T_file_path, 'r') as file:
            filedata = file.read()
        filedata = filedata.replace('uniform 473.15;', f'uniform {T_obst[i]};')
        filedata = filedata.replace('uniform 288.15;', f'uniform {T_inlet[i]};')

        with open(T_file_path, 'w') as file:
            file.write(filedata)

        # Modify the velocity in the 0/U file
        U_file_path = f"./{sim_data_path}/{i}/0/U"
        with open(U_file_path, 'r') as file:
            u_filedata = file.read()
            
        u_filedata = u_filedata.replace(
            'value           uniform (1 0 0);',
            f'value           uniform ({U_inlet[i]} 0 0);'
        )
        with open(U_file_path, 'w') as file:
            file.write(u_filedata)

        # Modify the write interval in the system/controlDict file
        controlDict_file_path = f"./{sim_data_path}/{i}/system/controlDict"
        with open(controlDict_file_path, 'r') as file:
            controlDict_data = file.read()
            
        controlDict_data = controlDict_data.replace(
            'writeInterval   0.25;',
            f'writeInterval   {delta_T_write[i]};'
        )
        controlDict_data = controlDict_data.replace(
            'endTime         1;',
            f'endTime         {end_time[i]};'
        )
        with open(controlDict_file_path, 'w') as file:
            file.write(controlDict_data)
