#/bin/bash

dropout_rate="0.1"

lr="5e-4"
batch_size="1024"

echo ${lr}

datasetPath="../generate_data_3d/dataset_plate_heat_3d_1sim_15ts_12apr_.hdf5"
outarrayPath="gridded_sim_data.h5"
outarrayFlatPath="PC_data.h5"

first_sim="0"
last_sim="1"
first_t="10"
num_ts="15"
n_samples_per_frame="10000"
var="0.95"

model_size="MLP_small"

num_epoch="1000"
chunk_size="1000"
grid_res="4e-3"
block_size="16"
max_num_PC="256"

train_3d --dataset_path ${datasetPath} --outarray_fn ${outarrayPath} \
   --first_sim ${first_sim} --last_sim ${last_sim} --last_t ${num_ts} --first_t ${first_t} \
   --lr ${lr} --n_samples_per_frame ${n_samples_per_frame} --var_p ${var} --var_in ${var} \
   --num_epoch ${num_epoch} --model_architecture ${model_size} \
   --outarray_flat_fn ${outarrayFlatPath} --dropout_rate ${dropout_rate} \
   --batch_size ${batch_size} --chunk_size ${chunk_size} --new_model "True" \
   --grid_res ${grid_res} --block_size ${block_size} --max_num_PC ${max_num_PC} 
