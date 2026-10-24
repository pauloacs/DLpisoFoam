import os
test_data_path = os.path.dirname(__file__)
# input data
array = os.path.join(test_data_path, 'array.npy')
y_bot = os.path.join(test_data_path, 'y_bot.npy')
y_top = os.path.join(test_data_path, 'y_top.npy')
z_bot = os.path.join(test_data_path, 'z_bot.npy')
z_top = os.path.join(test_data_path, 'z_top.npy')
obst = os.path.join(test_data_path, 'obst.npy')
#
tucker_factors_fn = os.path.join(test_data_path, 'tucker_factors.pkl')
maxs_fn = os.path.join(test_data_path, 'maxs')
PCA_std_vals_fn = os.path.join(test_data_path, 'mean_std.npz')
weights_fn = os.path.join(test_data_path, 'weights.h5')
