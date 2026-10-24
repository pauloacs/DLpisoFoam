To train a model, first generate a compatible dataset using the examples in
`classical/gen_datasets/generate_data_2d/`. Set `datasetPath` in
`run_train.sh` and `run_eval.sh` to the generated HDF5 file if it is not at the
default `../../gen_datasets/generate_data_2d/dataset_plate.hdf5` path.

Then, run the training script:
```
./run_train.sh
```

After training, you can evaluate the SM using your newly generated dataset:
```
./run_eval.sh
```
