#/bin/bash

datasetPath="../generate_data/dataset_plate.hdf5"
num_sims="1"
num_ts="4"
var="0.95"

model_name="model_MLP_small-std-0.95-drop0.02-lr0.0001-regNone-batch1024.h5"
echo ${model_name}
evaluation_script --model_name ${model_name} --var_p ${var} --var_in ${var} --dataset_path ${datasetPath} --n_sims ${num_sims} --n_ts ${num_ts} --save_plots
