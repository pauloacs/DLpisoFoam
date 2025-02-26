# Instructions for Generating a Training Dataset

1. **Create the Reference Simulation**: 
    - Set up your reference simulation named `original_snappy`.
    - This example uses a confined flow over a constant temperature plate.

2. **Generate Simulation Cases**:
    - Use the following command to generate N simulation cases:
      ```bash
      python make_dataset.py
      ```

3. **Run Simulations**:
    - Install the pisoFoam_write with 
      ```bash
      cd pisoFoam_write/
      wmake
      ```
    	(The pisoFoam_write was created to write all the necessary fields)
    - Execute all the simulations (under simulation_data/*) and store the flow fields as VTK files.


4. **Create HDF5 Dataset**:
    - Run the following script to gather data from all simulation cases and create an HDF5 dataset:
      ```bash
      ./dataset_gen.py
      ```


## Simulation details

# Domain
Domain L = 15 m
Height between plates H = 1 m
Distance from inlet to obstacle d = 4 m

# Obstacle - plate
plate length - L
angle of attack - alpha

