import os
test_data_path = os.path.dirname(__file__)
# input data
array = os.path.join(test_data_path, 'array.npy')
top = os.path.join(test_data_path, 'top.npy')
obst = os.path.join(test_data_path, 'obst.npy')
#
ipca_input_fn = os.path.join(test_data_path, 'ipca_input.pkl')
ipca_output_fn = os.path.join(test_data_path, 'ipca_output.pkl')
maxs_fn = os.path.join(test_data_path, 'maxs')
max_PCA_fn = os.path.join(test_data_path, 'maxs_PCA')
weights_fn = os.path.join(test_data_path, 'weights.h5')
