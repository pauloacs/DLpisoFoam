#/bin/bash

dropout_rate="0.02"

lr="1e-4"
batch_size="1024"

echo ${lr}

datasetPath="../generate_data/dataset_plate.hdf5"
outarrayPath="outarray.h5"
outarrayFlatPath="outarray_flat.h5"

num_sims="1"
num_ts="4"
n_samples="1e4"
var="0.95"

model_size="MLP_small"

num_epoch="200"
n_chunks="5"

train_script --dataset_path ${datasetPath} --outarray_fn ${outarrayPath} --num_sims ${num_sims} --num_epoch ${num_epoch} --lr ${lr} --n_samples ${n_samples} --var_p ${var} --var_in ${var} --model_architecture ${model_size} --last_t ${num_ts} --outarray_flat_fn ${outarrayFlatPath} --dropout_rate ${dropout_rate} --batch_size ${batch_size} --n_chunks ${n_chunks} --new_model "True"
