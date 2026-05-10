###################################################################################################
###################################################################################################
########################## STILL WORKING ON MAKING THIS WORK ######################################
###################################################################################################
###################################################################################################

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time
import traceback
import sys
import numpy as np

import mpi4py
mpi4py.rc.initialize = True
mpi4py.rc.finalize = False
from mpi4py import MPI

import pickle as pk
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import tensorly as tl

from pressure_SM._3D.CFD_usable.utils import memory
from pressure_SM._3D.train_and_eval.utils.data_processing import interpolate_fill, interp_weights, create_uniform_grid
from pressure_SM._3D.train_and_eval.utils.data_processing import interpolate_fill_njit
from pressure_SM._3D.train_and_eval.utils.domain_geometry import domain_dist
from pressure_SM._3D.train_and_eval.utils.model_utils import define_model_arch

from pressure_SM._3D.train_and_eval.assembly_vec_other import assemble_prediction
from pressure_SM._3D.train_and_eval.neural_networks import *


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
	ranks,
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

	global comm, rank, nprocs
	print('Initializing MPI communication in Python')
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	nprocs = comm.Get_size()

	if rank == 0:
		global in_factors, out_factors
		print('Loading the Tucker factors')
		with open(tucker_fn, 'rb') as f:
			tucker_data = pk.load(f)
		in_factors = tucker_data['input_factors']
		out_factors = tucker_data['output_factors']

		# ---- NEW: pretranspose factors for fast einsum paths (float32 + contiguous) ----
		# x_array has shape (N, bs, bs, bs, 4)
		# We need transpose=True for modes [1,2,3,4] => multiply by factor.T along each mode.
		global in_factors_T, out_factors_c
		in_factors_T = [None]  # keep 1-based indexing consistent with your existing list
		for k in range(1, 5):
			F = np.asarray(in_factors[k], dtype=np.float32)
			in_factors_T.append(np.ascontiguousarray(F.T))

		# For inverse Tucker (transpose=False), we want non-transposed factors contiguous.
		out_factors_c = [None]
		for k in range(1, 4):
			F = np.asarray(out_factors[k], dtype=np.float32)
			out_factors_c.append(np.ascontiguousarray(F))
		# ------------------------------------------------------------------------------


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
		input_features_size = ranks * ranks * ranks * 4
		output_features_size = ranks * ranks * ranks

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
	if 'comm' in globals() and comm.Get_size() > 1:
		array_global = comm.gather(array, root=0)
		obst_global = comm.gather(obst_boundary, root=0)
		y_bot_global = comm.gather(y_bot_boundary, root=0)
		z_bot_global = comm.gather(z_bot_boundary, root=0)
		y_top_global = comm.gather(y_top_boundary, root=0)
		z_top_global = comm.gather(z_top_boundary, root=0)
	else:
		array_global = [array]
		obst_global = [obst_boundary]
		y_bot_global = [y_bot_boundary]
		z_bot_global = [z_bot_boundary]
		y_top_global = [y_top_boundary]
		z_top_global = [z_top_boundary]

	global len_rankwise

	grid_res = grid_res_g

	len_rankwise = comm.gather(array.shape[0], root = 0)

	if rank == 0:

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

		limits = {
			'x_min': round(np.min(array_concat[...,3]), 2),
			'x_max': round(np.max(array_concat[...,3]), 2),
			'y_min': round(np.min(array_concat[...,4]), 2),
			'y_max': round(np.max(array_concat[...,4]), 2),
			'z_min': round(np.min(array_concat[...,5]), 2),
			'z_max': round(np.max(array_concat[...,5]), 2)
		}

		X0, Y0, Z0 = create_uniform_grid(limits, grid_res)
		xyz0 = np.concatenate((np.expand_dims(X0, axis=1), np.expand_dims(Y0, axis=1), np.expand_dims(Z0, axis=1)), axis=-1)
		points = array_concat[...,3:6] #coordinates

		#print( 'Calculating verts and weights' )
		vert_OFtoNP, weights_OFtoNP = interp_weights(points, xyz0, interp_method='IDW')
		assert np.all(np.isfinite(weights_OFtoNP)), "NaN values found in weights_OFtoNP"
		assert np.all(np.isfinite(vert_OFtoNP)), "NaN values found in vert_OFtoNP"

		global vert_OFtoNP_array, weights_OFtoNP_array
		vert_OFtoNP_array = vert_OFtoNP
		weights_OFtoNP_array = weights_OFtoNP

		vert_OFtoNP = list(vert_OFtoNP)
		weights_OFtoNP = list(weights_OFtoNP)

		vert_NPtoOF, weights_NPtoOF = interp_weights(xyz0, points, interp_method='IDW')
		assert np.all(np.isfinite(weights_NPtoOF)), "NaN values found in weights_NPtoOF"
		assert np.all(np.isfinite(vert_NPtoOF)), "NaN values found in vert_NPtoOF"

		global vert_NPtoOF_array, weights_NPtoOF_array
		vert_NPtoOF_array = vert_NPtoOF
		weights_NPtoOF_array = weights_NPtoOF

		#print( 'Calculating domain bool' )
		# You may need to update domain_dist to accept the new boundaries if needed
		boundaries = {
			'obst_boundary': obst,
			'y_bot_boundary': y_bot,
			'z_bot_boundary': z_bot,
			'y_top_boundary': y_top,
			'z_top_boundary': z_top
		}

		domain_bool, sdf = domain_dist(boundaries, xyz0, grid_res)

		grid_shape_z = int(round((limits['z_max']-limits['z_min'])/grid_res))
		grid_shape_y = int(round((limits['y_max']-limits['y_min'])/grid_res)) 
		grid_shape_x = int(round((limits['x_max']-limits['x_min'])/grid_res))

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
		delta_ux = array_concat[...,0] #values
		delta_ux_interp = interpolate_fill_njit(delta_ux, vert_OFtoNP_array, weights_OFtoNP_array) 

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

def py_func(array_in, U_max_norm):
	"""
	Method called at each simulation time step to compute the pressure field based on an input velocity field.

	Args:
		array_in (ndarray): Input velocity field.
		U_max_norm (float): Maximum normalized velocity.

	Returns:
		ndarray: Predicted pressure field.
	"""
	# Gathering all the inputs in 1 thread
	if 'comm' in globals() and comm.Get_size() > 1:
		array_global = comm.gather(array_in, root = 0)
	else:
		array_global = [array_in]
	
	block_size = block_size_g

	if rank == 0: #running all calculations at rank 0 
		if verbose_g:
			print('Starting call of SM py_func...')

		t0_py_func = time.time()

		array = np.concatenate(array_global)

		t0 = time.time()

		#np.save('array.npy', array)
		#U_max_norm = np.max( np.sqrt( np.square(array[...,0:1]) + np.square(array[...,1:2]) ) )

		deltaU = array[...,0:3]#.astype(np.float32)
		deltaU_prev = array[...,3:6]#.astype(np.float32)
		deltaP_prev = array[...,6]#.astype(np.float32)

		# check where the deltaU has changed in the last time step
		deltaU_changed = np.abs(deltaU - deltaU_prev).sum(axis=-1)
		deltaU_changed = deltaU_changed / deltaU_changed.max()

		# Normalize deltaU components by U_max_norm
		deltaUx_adim = deltaU[...,0:1]/U_max_norm 
		deltaUy_adim = deltaU[...,1:2]/U_max_norm
		deltaUz_adim = deltaU[...,2:3]/U_max_norm

		# deltaUx_adim = deltaU[...,0]/U_max_norm 
		# deltaUy_adim = deltaU[...,1]/U_max_norm
		# deltaUz_adim = deltaU[...,2]/U_max_norm


		if verbose_g: 
			print(f"Data pre-processing: {time.time()-t0} s")

		t0 = time.time()

		deltaUx_interp = interpolate_fill_njit(deltaUx_adim[:,0], vert_OFtoNP_array, weights_OFtoNP_array)
		deltaUy_interp = interpolate_fill_njit(deltaUy_adim[:,0], vert_OFtoNP_array, weights_OFtoNP_array)
		deltaUz_interp = interpolate_fill_njit(deltaUz_adim[:,0], vert_OFtoNP_array, weights_OFtoNP_array)

		deltaU_changed_interp = interpolate_fill_njit(deltaU_changed, vert_OFtoNP_array, weights_OFtoNP_array)
		deltaP_prev_interp = interpolate_fill_njit(deltaP_prev, vert_OFtoNP_array, weights_OFtoNP_array)

		if verbose_g:
			print( f"1st interpolation took: {time.time()-t0}\ s")

		t0 = time.time()

		grid = np.zeros((grid_shape_z, grid_shape_y, grid_shape_x, 4), dtype=np.float32)
		deltaP_prev_grid = np.zeros(shape=(grid_shape_z, grid_shape_y, grid_shape_x), dtype=np.float32)
		deltaU_change_grid = np.zeros(shape=(grid_shape_z, grid_shape_y, grid_shape_x), dtype=np.float32)

        # Pre-compute tuple indices once
		idx_i, idx_j, idx_k = indices[:, 0], indices[:, 1], indices[:, 2]

		# Stack all interpolated values and assign at once
		interp_stack = np.column_stack([deltaUx_interp, deltaUy_interp, deltaUz_interp])
		grid[idx_i, idx_j, idx_k, :3] = interp_stack
		grid[:, :, :, 3] = sdfunct[:, :, :, 0]

		# Direct assignment
		deltaP_prev_grid[idx_i, idx_j, idx_k] = deltaP_prev_interp
		deltaU_change_grid[idx_i, idx_j, idx_k] = deltaU_changed_interp

		## Rescale using broadcasting with normalization array
		norm_factors = np.array([max_abs_delta_Ux, max_abs_delta_Uy, max_abs_delta_Uz, max_abs_dist], dtype=np.float32)
		grid /= norm_factors[None, None, None, :]

		if verbose_g:
			print( f"Filling grid with shape {grid.shape} took: {time.time()-t0} s")

		t0 = time.time()

		# Setting any nan value to 0 to avoid issues
		grid[np.isnan(grid)] = 0

		x_list = []
		indices_list = []

		overlap = int(overlap_ratio_g*block_size)

		n_x = int(np.ceil((grid_shape_x - block_size)/(block_size - overlap )) ) + 1
		n_y = int(np.ceil((grid_shape_y - block_size)/(block_size - overlap )) ) + 1
		n_z = int(np.ceil((grid_shape_z - block_size)/(block_size - overlap )) ) + 1

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
					debug_mode = False
					if verbose_g and debug_mode:
						print(f"{(z_0,z_f, y_0,y_f, x_0,x_f)}")
						print(f"indices: {(i, j, n_x -1 - k)}")
					x_list.append(grid[z_0:z_f, y_0:y_f, x_0:x_f, 0:4])
					indices_list.append([i, j, n_x -1 - k])

		x_array = np.array(x_list)

		if verbose_g:
			print(f"Data extraction loop took: {time.time()-t0} s")

		t0 = time.time()
		N = x_array.shape[0]
		features = x_array.shape[3]

		#input_core = tl.tenalg.multi_mode_dot(x_array, in_factors[1:], modes=[1, 2, 3, 4], transpose=True)
		# ---- REPLACE tensorly multi_mode_dot with a single einsum ----
		# x_array:       (N, I, J, K, F) where I=J=K=block_size, F=4
		# in_factors_T:  A (R, I), B (R, J), C (R, K), D (R, F)
		# result core:   (N, R, R, R, R)
		A = in_factors_T[1]
		B = in_factors_T[2]
		C = in_factors_T[3]
		D = in_factors_T[4]

		# einsum does: core[n,a,b,c,d] = sum_{i,j,k,f} x[n,i,j,k,f]*A[a,i]*B[b,j]*C[c,k]*D[d,f]
		input_core = np.einsum("nijkf,ai,bj,ck,df->nabcd", x_array, A, B, C, D, optimize=True)

		input_transformed = input_core.reshape(N, -1)

		# Standardize input PCs
		x_input = (input_transformed - mean_in) / std_in

		if verbose_g:
			print(f"Tucker transformation : {time.time()-t0} s")

		t0 = time.time()

		# Calling the NN to predict the principal components (PC) of the pressure field:
		# PC_input -> PC_p (if necessary could be done in batches)

		# res_concat = model.predict(x_input, batch_size=32)
		res_concat = np.array(model(x_input))

		if verbose_g:
			print(f"Model prediction time : {time.time()-t0} s")

		# Tucker inverse transformation:
		# tensor_cores_deltaP -> deltaP
		t0 = time.time()

		# Getting the non-standerdized PCs
		res_concat = (res_concat * std_out) + mean_out

		# Perform inverse transformation using Tucker factors
		#res_concat = tl.tenalg.multi_mode_dot(res_concat, out_factors[1:], modes=[1, 2, 3], transpose=False)
		
		# Core should be (N, R, R, R)
		core = res_concat.reshape(input_core[..., 0].shape).astype(np.float32, copy=False)

		U1 = out_factors_c[1]  # (I, R)
		U2 = out_factors_c[2]  # (J, R)
		U3 = out_factors_c[3]  # (K, R)

		# out[n,i,j,k] = sum_{a,b,c} core[n,a,b,c] * U1[i,a] * U2[j,b] * U3[k,c]
		res_concat = np.einsum("nabc,ia,jb,kc->nijk", core, U1, U2, U3, optimize=True)

		if verbose_g:
			print(f"Tucker inverse transform : {time.time()-t0} s")

		t0 = time.time()
		# Redimensionalizing the predicted pressure field
		res_concat = res_concat * max_abs_delta_p * pow(U_max_norm, 2.0)

		number_of_nans = np.isnan(res_concat).sum()
		if verbose_g:
			print(f"Number of NaNs in res_concat before assembling: {number_of_nans}/{res_concat.size}")
		assert number_of_nans == 0, "NaN values found in res_concat before assembling."
		
		for i, x in enumerate(x_list):
			if (x[0,:,:,0] < 1e-5).all() and (x[0,:,:,1] < 1e-5).all():
				res_concat[i] = np.zeros((block_size, block_size, 1))

		if verbose_g:
			print(f"Processing before assembly : {time.time()-t0} s")

		t0 = time.time()

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

		if verbose_g:
			print(f"Assembly algorithm took: {time.time()-t0} s")

		t0 = time.time()
		# Rearrange the prediction into a 1D array that it can be sent to OF
		change_in_deltap = change_in_deltap[tuple(indices.T)]

		number_of_nans = np.isnan(change_in_deltap).sum()
		if verbose_g:
			print(f"Max and min of change_in_deltap before filtering: {np.nanmax(change_in_deltap)}, {np.nanmin(change_in_deltap)}")		
			print(f"Number of NaNs in change_in_deltap before filtering: {number_of_nans}/{change_in_deltap.size}")
			assert number_of_nans == 0, "NaN values found in change_in_deltap before filtering."
		    
			print(f"Flatenning array to send to OF and checking NANs took: {time.time()-t0} s")
		
		t0 = time.time()

		# Interpolation into the orginal grid
		change_in_deltap = interpolate_fill_njit(change_in_deltap, vert_NPtoOF_array, weights_NPtoOF_array)
		#p_interp[np.isnan(p_interp)] = 0

		p = deltaP_prev + change_in_deltap

		if verbose_g:
			print(f"Final Interpolation took: {time.time()-t0} s")

		init = 0

		# Dividing p into a list of n elements (consistent with the OF domain decomposition)
		# This is necessary to enable parallelization in OF
		p_rankwise = [] 

		# Check if len_rankwise matches the number of ranks and total length
		if len(len_rankwise) != nprocs:
			raise ValueError(f"len_rankwise ({len(len_rankwise)}) does not match number of ranks ({nprocs})")
		if sum(len_rankwise) != len(p):
			raise ValueError(f"Sum of len_rankwise ({sum(len_rankwise)}) does not match length of p ({len(p)})")

		for length in len_rankwise:
			end = init + length
			p_rankwise.append(p[init:end,...])
			init += length

		if verbose_g:
			print( f"The whole python function took : {time.time()-t0_py_func} s")

	else:
		p_rankwise = None

	for p_rank_i in p_rankwise:
		if np.any(np.isnan(p_rank_i)):
			nan_count = np.sum(np.isnan(p_rank_i))
			if verbose_g:
				print(f"Number of NaN values in p_rankwise: {nan_count}")
			raise ValueError("Warning: NaN values detected in p_rankwise before scattering.")

	# This scatters the value to each worker
	p = comm.scatter(p_rankwise, root=0)

	if np.any(np.isnan(p)):
		print(f"Warning: NaN values detected in p at rank {rank} after scattering.")

	if verbose_g:
		print(f"Process {rank} received object with shape {p.shape}")

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
