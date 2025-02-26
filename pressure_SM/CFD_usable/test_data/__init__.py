import os
test_data_path = os.path.dirname(__file__)
# input data
array = os.path.join(test_data_path, 'array.npy')
top = os.path.join(test_data_path, 'top.npy')
obst = os.path.join(test_data_path, 'obst.npy')
#
pca_input_fn = os.path.join(test_data_path, 'pca_in.pkl')
pca_output_fn = os.path.join(test_data_path, 'pca_p.pkl')
maxs_fn = os.path.join(test_data_path, 'maxs')
PCA_std_vals_fn = os.path.join(test_data_path, 'mean_std.npz')
weights_fn = os.path.join(test_data_path, 'weights.h5')
