import os
from tqdm import *
import subprocess
import random
import numpy as np

scale = 2
grading = 3

# Regarding the channel lengths, it is defined in the gen_blockMeshDict_plate
# Those are domain_length = 0.75, domain_height = 0.1


# These are plate lengths - L, b
# For reference, H (distance between walls) = 0.1
# Geometry variables
L = np.array([0.0375] * 10)
alpha = [70, 50, 80, 40, 60, 80, 50, 60, 70, 40]
b = [0.0025] * 10
x_cord = [0.2] * 10

# kinematic viscosity of water at 20ºC
nu = 1.053E-6

# OPTION 1
# Deriving Re from velocity
U_inlet =np.array([5, 20, 15, 20, 30, 40, 50, 70, 90, 100])
Re = (U_inlet * L / nu).round(0)

# OPTION 2
# Deriving velocity from Re
Re =np.array([500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000])
U_inlet = (Re * nu / L).round(3)

print(f'L: {list(L)}')
print(f'alpha: {list(alpha)}')
print(f'U_inlet: {list(U_inlet)}')
print(f'Re numbers: {list(Re)}')

# Sampling based on t* = L/U
#delta_T_write = np.round(np.array(L) / np.array(U_inlet), 4)

# Sampling based on t* = L_domain/U
# Makes sure the flow goes through the domain 1.5X
domain_L = 1
n_ts = 25
delta_T_write = np.round(np.array(domain_L * 1.5) / np.array(U_inlet) / n_ts, 4)
deltaT =  np.round(delta_T_write / 200, 4)
end_time = delta_T_write * n_ts

num_runs = len(L)
sim_data_path = 'simulation_data_1'

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
            'writeInterval   0.5;',
            f'writeInterval   {delta_T_write[i]};'
        )
        controlDict_data = controlDict_data.replace(
            'endTime         25;',
            f'endTime         {end_time[i]};'
        )
        controlDict_data = controlDict_data.replace(
            'deltaT          0.0002;',
            f'deltaT          {deltaT[i]};'
        )
        with open(controlDict_file_path, 'w') as file:
            file.write(controlDict_data)
