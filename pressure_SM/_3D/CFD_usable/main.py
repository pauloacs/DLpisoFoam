import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time
import traceback
import sys
import numpy as np

import pickle as pk
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import tensorly as tl

from pressure_SM._3D.CFD_usable.utils import memory
from pressure_SM._3D.train_and_eval.utils import interpolate_fill, interp_weights, domain_dist, create_uniform_grid, define_model_arch
from pressure_SM._3D.train_and_eval.assembly import assemble_prediction
from pressure_SM._3D.train_and_eval.neural_networks import *

########
# TODO: Find ways to optimize performance:
# Ideas:
# 1 - Use numba jit to optimize the code.
# 2 - Improve the reassembly method.
# 3 - Identify and fix any inefficiencies in the code.

## Typical running times 
## (from running python tests/test_module.py):

# Data pre-processing: 7.7009e-05 s
# 1st interpolation took: 0.0406 s
# 2nd interpolation took: 0.0285 s
# Reassembly algorithm took: 0.0242 s
# Applying Gaussian filter took: 4.7684e-07 s
# Final Interpolation took: 0.0017 s
# The whole python function took: 0.2002 s

#########

def load_tucker_and_NN(
	tucker_fn,
	maxs_fn,
	std_vals_fn,
	weights_fn,
	model_arch,
	apply_filter,
	overlap_ratio,
	filter_tuple,
	block_size,
	grid_res,
	dropout_rate,
	regularization,
	verbose=True
):
	"""
	Load tucker factors and initialize the trained neural network model.

	Parameters:
	maxs_fn (str): File path to the maximum values file.
    std_vals_fn (str): File path to the maximum feature values file.
	weights_fn (str): File path to the neural network model weights.
	model_arch (str): NN architecture.
	apply_filter (bool):
	overlap_ratio (float):
	verbose (bool, optional): Whether to print verbose output. Defaults to True.

	Returns:
	None
	"""
    # Set the global configuration
	global apply_filter_g, overlap_ratio_g, verbose_g, filter_tuple_g, block_size_g, grid_res_g
	apply_filter_g = apply_filter
	overlap_ratio_g = overlap_ratio
	verbose_g = verbose
	filter_tuple_g = filter_tuple
	block_size_g = block_size
	grid_res_g = grid_res

	global in_factors, out_factors
	print('Loading the Tucker factors')
	with open(tucker_fn, 'rb') as f:
		tucker_data = pk.load(f)
	in_factors = tucker_data['input_factors']
	out_factors = tucker_data['output_factors']

	## Loading values for blocks normalization
	maxs = np.loadtxt(maxs_fn)
	global max_abs_delta_Ux, max_abs_delta_Uy, max_abs_delta_Uz, max_abs_dist, max_abs_delta_p
	(max_abs_delta_Ux,
  		max_abs_delta_Uy,
		max_abs_delta_Uz,
		max_abs_dist,
		max_abs_delta_p) = maxs
	
	# Loading values for standardization
	data = np.load(std_vals_fn)
	global mean_in, std_in, mean_out, std_out
	mean_in = data['mean_in']
	std_in = data['std_in']
	mean_out = data['mean_out']
	std_out = data['std_out']

	print('Initializing NN')
	n_layers, width = define_model_arch(model_arch)

	global model
	input_features_size = 4 * 4 * 4 * 4
	output_features_size = 4 * 4 * 4

	model = MLP(n_layers, width, input_features_size, output_features_size, dropout_rate, regularization)
	model.load_weights(weights_fn)

	if verbose:
		print("Neural network initialized.")
		print(f"Model architecture: {model_arch}")
		print(f"overlap_ratio: {overlap_ratio}")


def init_func(array, z_top_boundary, z_bot_boundary, y_top_boundary, y_bot_boundary, obst_boundary):
	"""
	Initialization function for the simulation.

	This function is called at the beginning of a simulation to compute everything that is static, including interpolation weights and vertices.

	Args:
		array (ndarray): Ux, Uy, and coordinates at each mesh cell center.
		obst_boundary (ndarray): Obstacle boundary.
		y_bot_boundary (ndarray): Bottom boundary in y-direction.
		z_bot_boundary (ndarray): Bottom boundary in z-direction.
		y_top_boundary (ndarray): Top boundary in y-direction.
		z_top_boundary (ndarray): Top boundary in z-direction.

	Returns:
		int: Returns 0 after successful initialization.

	Notes:
		- This function may take a while to run.
		- The function gathers data from all ranks and performs computations on rank 0.
		- The function calculates interpolation weights and vertices for both OFtoNP and NPtoOF.
		- The function calculates the domain boolean and signed distance function.
		- The function initializes indices, sdfunct, vert_OFtoNP, weights_OFtoNP, vert_NPtoOF, weights_NPtoOF, shape_y, and shape_x.
	"""
	# if 'comm' in globals() and comm.Get_size() > 1:
	# 	array_global = comm.gather(array, root=0)
	# 	obst_global = comm.gather(obst_boundary, root=0)
	# 	y_bot_global = comm.gather(y_bot_boundary, root=0)
	# 	z_bot_global = comm.gather(z_bot_boundary, root=0)
	# 	y_top_global = comm.gather(y_top_boundary, root=0)
	# 	z_top_global = comm.gather(z_top_boundary, root=0)
	# else:
	array_global = [array]
	obst_global = [obst_boundary]
	y_bot_global = [y_bot_boundary]
	z_bot_global = [z_bot_boundary]
	y_top_global = [y_top_boundary]
	z_top_global = [z_top_boundary]

	grid_res = grid_res_g


	print('Running init function... This might take a while! ')
	array_concat = np.concatenate(array_global)
	obst = np.concatenate(obst_global)
	y_bot = np.concatenate(y_bot_global)
	z_bot = np.concatenate(z_bot_global)
	y_top = np.concatenate(y_top_global)
	z_top = np.concatenate(z_top_global)

	#for debugging purposes
	# np.save('obst.npy', obst)
	# np.save('array.npy', array_concat)
	# np.save('y_bot.npy', y_bot)
	# np.save('z_bot.npy', z_bot)
	# np.save('y_top.npy', y_top)
	# np.save('z_top.npy', z_top)
	
	global indices, sdfunct
	global vert_OFtoNP, weights_OFtoNP, vert_NPtoOF, weights_NPtoOF
	global grid_shape_z, grid_shape_y, grid_shape_x

	x_min = round(np.min(array_concat[...,3]), 6)
	x_max = round(np.max(array_concat[...,3]), 6)
	y_min = round(np.min(array_concat[...,4]), 6)
	y_max = round(np.max(array_concat[...,4]), 6)
	z_min = round(np.min(array_concat[...,5]), 6)
	z_max = round(np.max(array_concat[...,5]), 6)

	X0, Y0, Z0 = create_uniform_grid(x_min, x_max, y_min, y_max, z_min, z_max, grid_res)
	xyz0 = np.concatenate((np.expand_dims(X0, axis=1), np.expand_dims(Y0, axis=1), np.expand_dims(Z0, axis=1)), axis=-1)
	points = array_concat[...,3:6] #coordinates

	#print( 'Calculating verts and weights' )
	vert_OFtoNP, weights_OFtoNP = interp_weights(points, xyz0)
	vert_NPtoOF, weights_NPtoOF = interp_weights(xyz0, points)

	#print( 'Calculating domain bool' )
	# You may need to update domain_dist to accept the new boundaries if needed
	boundaries_list = [obst, y_bot, z_bot, y_top, z_top]
	domain_bool, sdf = domain_dist(boundaries_list, xyz0, grid_res, find_limited_index=False)

	grid_shape_z = int(round((z_max-z_min)/grid_res))
	grid_shape_y = int(round((y_max-y_min)/grid_res)) 
	grid_shape_x = int(round((x_max-x_min)/grid_res))

	x0 = np.min(X0)
	y0 = np.min(Y0)
	z0 = np.min(Z0)
	dx = grid_res
	dy = grid_res
	dz = grid_res

	indices= np.empty((X0.shape[0], 3))
	obst_bool = np.zeros((grid_shape_z, grid_shape_y, grid_shape_x, 1))
	sdfunct = obst_bool.copy()

	#to compute bool 
	delta_ux = array_concat[...,0:1] #values
	delta_ux_interp = interpolate_fill(delta_ux, vert_OFtoNP, weights_OFtoNP) 

	for (step, x_y_z) in enumerate(xyz0):
		if domain_bool[step] * (~np.isnan(delta_ux_interp[step])):
			ii = int(round((x_y_z[..., 2] - z0) / dz))
			jj = int(round((x_y_z[..., 1] - y0) / dy))
			kk = int(round((x_y_z[..., 0] - x0) / dx))
			indices[step, 0] = ii
			indices[step, 1] = jj
			indices[step, 2] = kk
			sdfunct[ii, jj, kk, :] = sdf[step]
			obst_bool[ii, jj, kk, :] = int(1)

	indices = indices.astype(int)
	print('Init function ran successfully! :D')
	#sys.stdout.flush()

	return 0

def py_func(array_in, U_max_norm, verbose=False):
	"""
	Method called at each simulation time step to compute the pressure field based on an input velocity field.

	Args:
		array_in (ndarray): Input velocity field.
		U_max_norm (float): Maximum normalized velocity.

	Returns:
		ndarray: Predicted pressure field.
	"""
	# Gathering all the inputs in 1 thread
	#if 'comm' in globals() and comm.Get_size() > 1:
	#	array_global = comm.gather(array_in, root = 0)
	#else:
	
	array_global = [array_in]
	
	if verbose_g:
		print('Starting call of SM')

	block_size = block_size_g

	t0_py_func = time.time()

	array = np.concatenate(array_global)

	t0 = time.time()

	#np.save('array.npy', array)
	#U_max_norm = np.max( np.sqrt( np.square(array[...,0:1]) + np.square(array[...,1:2]) ) )

	deltaU = array[...,0:3]
	deltaU_prev = array[...,3:6]
	deltaP_prev = array[...,6]

	# check where the deltaU has changed in the last time step
	deltaU_changed = np.abs(deltaU - deltaU_prev).sum(axis=-1)
	deltaU_changed = deltaU_changed / deltaU_changed.max()
	# Normalize deltaU components by U_max_norm
	deltaUx_adim = deltaU[...,0:1]/U_max_norm 
	deltaUy_adim = deltaU[...,1:2]/U_max_norm
	deltaUz_adim = deltaU[...,2:3]/U_max_norm

	t1 = time.time()
	if verbose_g: 
		print( "Data pre-processing:" + str(t1-t0) + " s")

	t0 = time.time()

	# Interpolate all three velocity components
	deltaUx_interp = interpolate_fill(deltaUx_adim, vert_OFtoNP, weights_OFtoNP)
	deltaUy_interp = interpolate_fill(deltaUy_adim, vert_OFtoNP, weights_OFtoNP)
	deltaUz_interp = interpolate_fill(deltaUz_adim, vert_OFtoNP, weights_OFtoNP)
	
	deltaU_changed_interp = interpolate_fill(deltaU_changed, vert_OFtoNP, weights_OFtoNP)
	deltaP_prev_interp = interpolate_fill(deltaP_prev, vert_OFtoNP, weights_OFtoNP)

	t1 = time.time()
	if verbose_g:
		print( "1st interpolation took:" + str(t1-t0) + " s")

	t0 = time.time()

	grid = np.zeros((grid_shape_z, grid_shape_y, grid_shape_x, 4), dtype=np.float32)

	# Rearrange interpolated 1D arrays into 3D arrays
	grid[:, :, :, 0:1][tuple(indices.T)] = deltaUx_interp.reshape(deltaUx_interp.shape[0], 1)
	grid[:, :, :, 1:2][tuple(indices.T)] = deltaUy_interp.reshape(deltaUy_interp.shape[0], 1)
	grid[:, :, :, 2:3][tuple(indices.T)] = deltaUz_interp.reshape(deltaUz_interp.shape[0], 1)

	grid[:, :, :, 3:4] = sdfunct

	deltaP_prev_grid = np.zeros(shape=(grid_shape_z, grid_shape_y, grid_shape_x))
	deltaU_change_grid = np.zeros(shape=(grid_shape_z, grid_shape_y, grid_shape_x))

	deltaP_prev_grid[tuple(indices.T)] = deltaP_prev_interp.reshape(deltaP_prev_interp.shape[0])
	deltaU_change_grid[tuple(indices.T)] = deltaU_changed_interp.reshape(deltaU_changed_interp.shape[0])

	## Rescale input variables to [-1,1]
	grid[:,:,:,0:1] = grid[0,:,:,0:1] / max_abs_delta_Ux
	grid[:,:,:,1:2] = grid[0,:,:,1:2] / max_abs_delta_Uy
	grid[:,:,:,2:3] = grid[0,:,:,2:3] / max_abs_delta_Uz
	grid[:,:,:,3:4] = grid[0,:,:,3:4] / max_abs_dist

	t1 = time.time()
	if verbose_g:
		print( "2nd interpolation took:" + str(t1-t0) + " s")

	# Setting any nan value to 0 to avoid issues
	grid[np.isnan(grid)] = 0

	x_list = []
	indices_list = []

	overlap = int(overlap_ratio_g*block_size)

	n_x = int(np.ceil((grid_shape_x - block_size)/(block_size - overlap )) ) + 1
	n_y = int(np.ceil((grid_shape_y - block_size)/(block_size - overlap )) ) + 1
	n_z = int(np.ceil((grid_shape_z - block_size)/(block_size - overlap )) ) + 1

	t0 = time.time()

	# Sampling blocks of size [block_size X block_size] from the input fields (Ux, Uy and sdf)
	# In the indices_list, the indices corresponding to each sampled block is stored to enable domain reconstruction later.
	for i in range (n_z):
		z_0 = i*block_size - i*overlap
		if i == n_z - 1:
			z_0 = grid_shape_z - block_size
		z_f = z_0 + block_size
		for j in range (n_y):
			y_0 = j*block_size - j*overlap
			if j == n_y - 1:
				y_0 = grid_shape_y - block_size
			y_f = y_0 + block_size
			for k in range(n_x):
				# going right to left
				x_0 = grid_shape_x - k*block_size + k*overlap - block_size
				if k == n_x - 1:
					x_0 = 0
				x_f = x_0 + block_size

				#DEBUGGING print
				#print(f"{(z_0,z_f, y_0,y_f, x_0,x_f)}")
				#print(f"indices: {(i, j, n_x -1 - k)}")
				x_list.append(grid[z_0:z_f, y_0:y_f, x_0:x_f, 0:4])
				indices_list.append([i, j, n_x -1 - k])

	x_array = np.array(x_list)

	t1 = time.time()
	if verbose_g:
		print( "Data extraction loop took:" + str(t1-t0) + " s")


	t0 = time.time()
	N = x_array.shape[0]
	features = x_array.shape[3]

	#x_array_flat = x_array.reshape((N, grid_shape_z * grid_shape_y * grid_shape_x, features ))
	#input_flat = x_array_flat.reshape((x_array_flat.shape[0],-1))

	#input_transformed = pcainput.transform(input_flat)[:,:PC_input]
	input_core = tl.tenalg.multi_mode_dot(x_array, in_factors[1:], modes=[1, 2, 3, 4], transpose=True)
	input_transformed = input_core.reshape(N, -1)

	# Standardize input PCs
	x_input = (input_transformed - mean_in) / std_in

	t1 = time.time()
	if verbose_g:
		print( "Tucker transformation : " + str(t1-t0) + " s")

	t0 = time.time()

	# Calling the NN to predict the principal components (PC) of the pressure field:
	# PC_input -> PC_p (if necessary could be done in batches)

	res_concat = np.array(model(x_input))
	t1 = time.time()
	if verbose_g:
		print( "Model prediction time : " + str(t1-t0) + " s")

	# PCA inverse transformation:
	# PC_deltaP -> deltaP
	t0 = time.time()

	# Getting the non-standerdized PCs
	res_concat = (res_concat * std_out) + mean_out
	res_concat = res_concat.reshape(input_core[...,0].shape)

	# Perform inverse transformation using Tucker factors
	res_concat = tl.tenalg.multi_mode_dot(res_concat, out_factors[1:], modes=[1, 2, 3], transpose=False)
	t1 = time.time()
	if verbose_g:
		print( "PCA inverse transform : " + str(t1-t0) + " s")

	# Redimensionalizing the predicted pressure field
	res_concat = res_concat * max_abs_delta_p * pow(U_max_norm, 2.0)

	for i, x in enumerate(x_list):
		if (x[0,:,:,0] < 1e-5).all() and (x[0,:,:,1] < 1e-5).all():
			res_concat[i] = np.zeros((block_size, block_size, 1))

	# The boundary condition is a fixed pressure of 0 at the output
	Ref_BC = 0 

	deltap_res, change_in_deltap = assemble_prediction(
		res_concat,
		indices_list,
		n_x,
		n_y,
		n_z,
		overlap,
		block_size,
		Ref_BC,
		x_array,
		apply_filter_g,
		grid_shape_x,
		grid_shape_y,
		grid_shape_z,
		deltaU_change_grid,
		deltaP_prev_grid,
		True,
	)

	t1 = time.time()
	if verbose_g:
		print( "Reassembly algorithm took:" + str(t1-t0) + " s")

	# Rearrange the prediction into a 1D array that it can be sent to OF
	change_in_deltap = change_in_deltap[tuple(indices.T)]

	t0 = time.time()
	# Interpolation into the orginal grid
	# Takes virtually no time because "vert" and "weigths" where already calculated on the init_func
	change_in_deltap = interpolate_fill(change_in_deltap, vert_NPtoOF, weights_NPtoOF)
	#p_interp[np.isnan(p_interp)] = 0
	t1 = time.time()
	if verbose_g:
		print( "Final Interpolation took:" + str(t1-t0) + " s")

	p = deltaP_prev + change_in_deltap

	t1_py_func = time.time()
	if verbose_g:
		print( "The whole python function took : " + str(t1_py_func-t0_py_func) + " s")


	return p

if __name__ == '__main__':
    print('This is the Python module for DLPoissonFOam')

	# Debugging code 

	#import h5py 
	#from numba import njit

	#@njit
	#def index(array, item):
	#    for idx, val in np.ndenumerate(array):
	#        if val == item:
	#            return idx
	#    # If no item was found return None, other return types might be a problem due to
	#    # numbas type inference.


	##path = '/home/paulo/dataset_unsteadyCil_fu_bound.hdf5' #adjust path

	##frame = 40
	##hdf5_file = h5py.File(path, "r")
	###data = hdf5_file["sim_data"][:1, frame-1:frame, ...]
	##top_boundary = hdf5_file["top_bound"][0, frame, ...]
	##obst_boundary = hdf5_file["obst_bound"][0, frame, ...]
	##hdf5_file.close()

	##indice_top = index(top_boundary[:,0] , -100.0 )[0]
	##top_boundary = top_boundary[:indice_top,:]

	##indice_obst = index(obst_boundary[:,0] , -100.0 )[0]
	##obst_boundary = obst_boundary[:indice_obst,:]

	##indice = index(data[0,0,:,0] , -100.0 )[0]
	##array = data[0,0,:indice,:4]
	##array[:,2:4] = data[0,0,:indice,3:5]

	#array = np.load('array.npy')
	#top_boundary = np.load('top.npy')
	#obst_boundary = np.load('obst.npy')

	#print(array.shape)

	#plt.scatter(array[:,2],array[:,3], c = array[:,0])
	#plt.show()

	#plt.scatter(array[:,2],array[:,3], c = array[:,1])
	#plt.show()

	#init_func(array, top_boundary, obst_boundary)
	#p = py_func(array)

	#plt.scatter(array[:,2], array[:,3], c=p, cmap = 'jet')
	#plt.show()
