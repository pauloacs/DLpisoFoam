#/bin/bash

datasetPath="../generate_data_3d/dataset_plate_heat_3d_1sim_15ts_12apr_.hdf5"
first_sim="0"
last_sim="1"
first_t="10"
last_t="15"
var="0.95"
delta="4e-3"
block_size="16"
overlap_ratio="0.7"

model_name="model_MLP_small-std-0.95-drop0.1-lr0.0005-regNone-batch1024.h5"
echo ${model_name}
eval_3d --model_name ${model_name} --var_p ${var} --var_in ${var} --dataset_path ${datasetPath} \
   --first_sim ${first_sim} --last_sim ${last_sim} --first_t ${first_t} --last_t ${last_t} \
   --delta ${delta} --block_size ${block_size} --save_plots --overlap_ratio ${overlap_ratio}
