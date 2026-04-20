import time
import traceback
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

import mpi4py
mpi4py.rc.initialize = True
mpi4py.rc.finalize = False
from mpi4py import MPI

import pickle as pk
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

from pressure_SM._2D.CFD_usable.utils import *
from pressure_SM._2D.CFD_usable.utils import interpolate_fill_njit

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

def load_pca_and_NN(ipca_input_fn, ipca_output_fn, maxs_fn, PCA_std_vals_fn, weights_fn, var, model_arch, apply_filter, overlap_ratio, filter_tuple, block_size=128, grid_res=5e-3, verbose=True):
	"""
	Load PCA mapping and initialize the trained neural network model.

	Parameters:
	ipca_input_fn (str): File path to the input PCA model.
	ipca_output_fn (str): File path to the output PCA model.
	maxs_fn (str): File path to the maximum values file.
    PCA_std_vals_fn (str): File path to the maximum PCA values file.
	weights_fn (str): File path to the neural network model weights.
	var (float): Variance to be retained.
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

	print('Loading the PCA mapping')
	pcainput = pk.load(open(ipca_input_fn, 'rb'))
	pcap = pk.load(open(ipca_output_fn, 'rb'))

	## Loading values for blocks normalization
	maxs = np.loadtxt(maxs_fn)
	global max_abs_ux, max_abs_uy, max_abs_dist, max_abs_p
	max_abs_ux, max_abs_uy, max_abs_dist, max_abs_p = maxs

	# Loading values for PCA standardization
	data = np.load(PCA_std_vals_fn)
	global mean_in, std_in, mean_out, std_out
	mean_in = data['mean_in']
	std_in = data['std_in']
	mean_out = data['mean_out']
	std_out = data['std_out']

	# Selecting the number of PCs to be used based on the variance to be retained
	PC_p = int(np.argmax(pcap.explained_variance_ratio_.cumsum() > var))
	PC_input = int(np.argmax(pcainput.explained_variance_ratio_.cumsum() > var))

	# Saving the PCA mapping
	global comp_p, pca_mean_p, comp_input, pca_mean_input, model
	comp_p = pcap.components_[:PC_p, :]
	pca_mean_p = pcap.mean_
	# pretranspose + cast to float32 (avoids float64 upcasting at each timestep)
	global comp_p_T, pca_mean_p_f32
	comp_p_T = np.ascontiguousarray(comp_p, dtype=np.float32)
	pca_mean_p_f32 = np.ascontiguousarray(pcap.mean_, dtype=np.float32)
	comp_input = pcainput.components_[:PC_input,:]
	pca_mean_input = pcainput.mean_
	# pretranspose + cast to float32
	global comp_input_T, pca_mean_input_f32
	comp_input_T = np.ascontiguousarray(comp_input.T, dtype=np.float32)
	pca_mean_input_f32 = np.ascontiguousarray(pcainput.mean_, dtype=np.float32)

	print('Initializing NN')
	if model_arch == 'MLP_small':
		n_layers = 3
		width = [512]*3
	elif model_arch == 'small_unet':
		n_layers = 9
		width = [512, 256, 128, 64, 32, 64, 128, 256, 512]
	elif model_arch == 'conv1D':
		n_layers = 7
		width = [128, 64, 32, 16, 32, 64, 128]
		convNN = True
	elif model_arch == 'MLP_medium':
		n_layers = 5
		width = [256] + [512]*3 + [256]
	elif model_arch == 'MLP_big':
		n_layers = 7
		width = [256] + [512]*5 + [256]
	elif model_arch == 'MLP_huge':
		n_layers = 12
		width = [256] + [512]*10 + [256]
	else:
		raise ValueError('Invalid NN model type')
	model = densePCA(PC_input, PC_p, n_layers, width)
	#model.load_weights(weights_fn)

	global comm, rank, nprocs
	print('Initializing MPI communication in Python')
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	nprocs = comm.Get_size()

	if verbose:
		print("PCA models and neural network initialized.")
		print(f"Variance retained: {var}")
		print(f"Model architecture: {model_arch}")
		print(f"overlap_ratio: {overlap_ratio}")


def init_func(array, top_boundary, obst_boundary):
	"""
	Initialization function for the simulation.

	This function is called at the beginning of a simulation to compute everything that is static, including interpolation weights and vertices.

	Args:
		array (ndarray): Ux, Uy, and coordinates at each mesh cell center.
		top_boundary (ndarray): Top boundary.
		obst_boundary (ndarray): Obstacle boundary.

	Returns:
		int: Returns 0 after successful initialization.

	Notes:
		- This function may take a while to run.
		- The function gathers data from all ranks and performs computations on rank 0.
		- The function calculates interpolation weights and vertices for both OFtoNP and NPtoOF.
		- The function calculates the domain boolean and signed distance function.
		- The function initializes indices, sdfunct, vert_OFtoNP, weights_OFtoNP, vert_NPtoOF, weights_NPtoOF, shape_y, and shape_x.
	"""
	
	array_global = comm.gather(array, root = 0)
	top_global = comm.gather(top_boundary, root = 0)
	obst_global = comm.gather(obst_boundary, root = 0)

	global len_rankwise

	len_rankwise = comm.gather(array.shape[0], root = 0)

	if rank == 0:

		print('Running init function... This might take a while! ')
		array_concat = np.concatenate(array_global)
		top = np.concatenate(top_global)
		obst = np.concatenate(obst_global)

		#np.save('top.npy', top)
		#np.save('obst.npy', obst)
		#np.save('array.npy', array_concat)
		
		global indices, sdfunct, vert_OFtoNP, weights_OFtoNP, vert_NPtoOF, weights_NPtoOF, shape_y, shape_x

		# Uniform grid resolution
		delta = grid_res_g

		x_min = round(np.min(array_concat[...,2]),3)
		x_max = round(np.max(array_concat[...,2]),3)

		y_min = round(np.min(array_concat[...,3]),3)
		y_max = round(np.max(array_concat[...,3]),3)

		domain_limits = [x_min, x_max, y_min, y_max]

		X0, Y0 = create_uniform_grid(x_min, x_max, y_min, y_max, delta)

		xy0 = np.concatenate((np.expand_dims(X0, axis=1),np.expand_dims(Y0, axis=1)), axis=-1)
		points = array_concat[...,2:4] #coordinates

		#print( 'Calculating verts and weights' )
		vert_OFtoNP, weights_OFtoNP = interp_barycentric_weights(points, xy0) #takes ~100% of the interpolation cost and it's only done once for each different mesh/simulation case
		vert_NPtoOF, weights_NPtoOF = interp_barycentric_weights(xy0, points) #takes ~100% of the interpolation cost and it's only done once for each different mesh/simulation case
	
		#print( 'Calculating domain bool' )
		domain_bool, sdf = domain_dist(top, obst, xy0, domain_limits)

		shape_y = int(round((y_max-y_min)/delta))
		shape_x = int(round((x_max-x_min)/delta))

		x0 = np.min(X0)
		y0 = np.min(Y0)
		dx = delta
		dy = delta

		indices= np.empty((X0.shape[0],2))
		obst_bool = np.zeros((shape_y,shape_x,1))
		sdfunct = np.zeros((shape_y,shape_x,1))

		#to compute bool 
		ux = array_concat[...,0:1] #values
		ux_interp = interpolate_fill(ux, vert_OFtoNP, weights_OFtoNP) 

		for (step, x_y) in enumerate(xy0):  
			if domain_bool[step] * (~np.isnan(ux_interp[step])) :
				jj = int(round((x_y[...,0] - x0) / dx))
				ii = int(round((x_y[...,1] - y0) / dy))

				indices[step,0] = ii
				indices[step,1] = jj
				sdfunct[ii,jj,:] = sdf[step]
				obst_bool[ii,jj,:] = int(1)

		indices = indices.astype(int)
		# cache int32 index arrays for fast grid assignment each timestep
		global indices_i, indices_j
		indices_i = indices[:, 0].astype(np.int32)
		indices_j = indices[:, 1].astype(np.int32)
		# cast interp weights to float32 so njit runs in float32 (2x SIMD width)
		global weights_OFtoNP_f32, vert_OFtoNP_i32, weights_NPtoOF_f32, vert_NPtoOF_i32
		weights_OFtoNP_f32 = np.ascontiguousarray(weights_OFtoNP, dtype=np.float32)
		vert_OFtoNP_i32 = np.ascontiguousarray(vert_OFtoNP, dtype=np.int32)
		weights_NPtoOF_f32 = np.ascontiguousarray(weights_NPtoOF, dtype=np.float32)
		vert_NPtoOF_i32 = np.ascontiguousarray(vert_NPtoOF, dtype=np.int32)
		# preallocate fixed-shape arrays reused every timestep
		global grid_buf, deltaP_prev_grid_buf, deltaU_change_grid_buf
		grid_buf = np.zeros((1, shape_y, shape_x, 3), dtype=np.float32)
		grid_buf[0, :, :, 2] = sdfunct[:, :, 0]   # SDF is static — set once
		deltaP_prev_grid_buf = np.zeros((shape_y, shape_x), dtype=np.float32)
		deltaU_change_grid_buf = np.zeros((shape_y, shape_x), dtype=np.float32)
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
	array_global = comm.gather(array_in, root = 0)

	if rank == 0: #running all calculations at rank 0 

		t0_py_func = time.time()

		array = np.concatenate(array_global)

		t0 = time.time()

		#np.save('array.npy', array)
		#U_max_norm = np.max( np.sqrt( np.square(array[...,0:1]) + np.square(array[...,1:2]) ) )

		deltaU = array[...,0:2]
		deltaU_prev = array[...,4:6]
		deltaP_prev = array[...,6]

		# check where the deltaU has changed in the last time step
		deltaU_changed = np.abs(deltaU - deltaU_prev).sum(axis=-1)
		deltaU_changed = deltaU_changed / deltaU_changed.max()

		deltaUx_adim = array[...,0:1]/U_max_norm 
		deltaUy_adim = array[...,1:2]/U_max_norm

		t1 = time.time()
		if verbose_g: 
			print( "Data pre-processing:" + str(t1-t0) + " s")

		t0 = time.time()

		# float32 njit interpolation (float32 weights give 2x SIMD width vs float64)
		deltaUx_interp = interpolate_fill_njit(deltaUx_adim.ravel().astype(np.float32), vert_OFtoNP_i32, weights_OFtoNP_f32)
		deltaUy_interp = interpolate_fill_njit(deltaUy_adim.ravel().astype(np.float32), vert_OFtoNP_i32, weights_OFtoNP_f32)
		deltaU_changed_interp = interpolate_fill_njit(deltaU_changed.astype(np.float32), vert_OFtoNP_i32, weights_OFtoNP_f32)
		deltaP_prev_interp = interpolate_fill_njit(deltaP_prev.astype(np.float32), vert_OFtoNP_i32, weights_OFtoNP_f32)

		t1 = time.time()
		if verbose_g:
			print( "1st interpolation took:" + str(t1-t0) + " s")

		t0 = time.time()
		# reuse preallocated buffers (no alloc, no SDF refill — set once at init)
		grid_buf[0, indices_i, indices_j, 0] = deltaUx_interp
		grid_buf[0, indices_i, indices_j, 1] = deltaUy_interp
		deltaP_prev_grid_buf[indices_i, indices_j] = deltaP_prev_interp
		deltaU_change_grid_buf[indices_i, indices_j] = deltaU_changed_interp
		# alias for readability
		grid = grid_buf
		deltaP_prev_grid = deltaP_prev_grid_buf
		deltaU_change_grid = deltaU_change_grid_buf

		## Rescale input variables to [-1,1] (in-place broadcast divide)
		grid[0] /= np.array([max_abs_ux, max_abs_uy, max_abs_dist], dtype=np.float32)

		t1 = time.time()
		if verbose_g:
			print( "2nd interpolation took:" + str(t1-t0) + " s")

		# Setting any nan value to 0 to avoid issues
		grid[np.isnan(grid)] = 0

		shape = block_size_g
		overlap = int(overlap_ratio_g*shape)

		n_x = int(np.ceil((shape_x-shape)/(shape - overlap )) )
		n_y = int((shape_y-shape)/(shape - overlap ))

		t0 = time.time()

		# CHANGED: preallocate arrays to avoid Python list appends + concatenate
		total_blocks = (n_y + 2) * (n_x + 1)
		x_array = np.empty((total_blocks, shape, shape, 3), dtype=np.float32)
		indices_list = np.empty((total_blocks, 2), dtype=np.int32)
		b = 0
		for i in range(n_y + 2):
			for j in range(n_x + 1):
				x_0 = grid.shape[2] - j*shape + j*overlap - shape
				if j == n_x: x_0 = 0
				x_f = x_0 + shape
				y_0 = i*shape - i*overlap
				if i == n_y + 1: y_0 = grid.shape[1]-shape
				y_f = y_0 + shape
				x_array[b] = grid[0, y_0:y_f, x_0:x_f, 0:3]
				indices_list[b] = [i, n_x - j]
				b += 1

		t1 = time.time()
		if verbose_g:
			print( "Data extraction loop took:" + str(t1-t0) + " s")

		t0 = time.time()
		N = x_array.shape[0]

		# single reshape; float32 mean avoids upcasting to float64
		input_flat = x_array.reshape(N, -1)
		input_transformed = np.dot(input_flat - pca_mean_input_f32, comp_input_T)

		# Standardize input PCs (in-place to avoid copies)
		input_transformed -= mean_in
		input_transformed /= std_in
		x_input = input_transformed

		t1 = time.time()
		if verbose_g:
			print( "PCA transform : " + str(t1-t0) + " s")

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

		# Getting the non-standerdized PCs (in-place to avoid copies)
		res_concat *= std_out
		res_concat += mean_out

		res_flat_inv = np.dot(res_concat, comp_p_T) + pca_mean_p_f32
		res_concat = res_flat_inv.reshape(N, shape, shape)
		t1 = time.time()
		if verbose_g:
			print( "PCA inverse transform : " + str(t1-t0) + " s")

		# Redimensionalizing the predicted pressure field (in-place)
		res_concat *= max_abs_p * pow(U_max_norm, 2.0)

		# CHANGED: vectorized zero-block detection (avoid Python loop)
		zero_blocks = (np.abs(x_array[:, :, :, 0]) < 1e-5).all(axis=(1, 2)) & \
		              (np.abs(x_array[:, :, :, 1]) < 1e-5).all(axis=(1, 2))
		res_concat[zero_blocks] = 0.0

		# The boundary condition is a fixed pressure of 0 at the output
		Ref_BC = 0

		## Domain reassembly method
		result = np.empty((shape_y, shape_x), dtype=np.float32)

		## Array to store the average pressure in the overlap region with the next down block
		BC_ups = np.zeros(n_x+1)

		# i index where the lower blocks are located
		p_i = shape_y - ( (shape-overlap) * n_y + shape )
		
		# j index where the left-most blocks are located
		p_j = shape_x - ( (shape - overlap) * n_x + shape )

		t0 = time.time()

		for i in range(len(x_array)):

			idx_i, idx_j = indices_list[i]
			flow_bool = x_array[i,:,:,2]
			pred_field = res_concat[i,:,:]
			## FIRST row
			if idx_i == 0:

				## Calculating correction to be applied
				if i == 0: 
 					## First correction - based on the outlet fixed pressure boundary condition
					BC_coor = np.mean(pred_field[:,-1][flow_bool[:,-1]!=0]) - Ref_BC
					#BC_coor = np.mean(pred_field[:,-1][flow_bool[:,-1]!=0]) - Ref_BC  # i = 0 sits outside the inclusion zone
				else:
					BC_ant_0 = np.mean(old_pred_field[:,:overlap][flow_bool[:,:overlap] !=0]) 
					BC_coor = np.mean(pred_field[:,-overlap:][flow_bool[:,-overlap:]!=0]) - BC_ant_0
				if idx_j == 0:
					intersect_zone_limit = overlap - p_j
					BC_ant_0 = np.mean(old_pred_field[:, :intersect_zone_limit][flow_bool[:, :intersect_zone_limit] !=0]) 
					BC_coor = np.mean(pred_field[:, -intersect_zone_limit:][flow_bool[:, -intersect_zone_limit:]!=0]) - BC_ant_0

				## Applying correction
				pred_field -= BC_coor

				## Storing upward average pressure
				BC_ups[idx_j] = np.mean(pred_field[-overlap:,:][flow_bool[-overlap:,:] !=0])

			## MIDDLE rows
			elif idx_i != n_y + 1:

				## Calculating correction to be applied
				if np.isnan(BC_ups[idx_j]): #### THIS PART IS NOT WORKING WELL ... CORRECT THIS!!
					if idx_j == 0:
						intersect_zone_limit = overlap - p_j
						BC_ant_0 = np.mean(old_pred_field[:, :intersect_zone_limit][flow_bool[:, :intersect_zone_limit] !=0]) 
						BC_coor = np.mean(pred_field[:, -intersect_zone_limit:][flow_bool[:, -intersect_zone_limit:]!=0]) - BC_ant_0
					elif idx_j == n_x:
						# Here it always needs to be corrected from above to keep consistency
						BC_ant_0 = np.mean(old_pred_field[:,:overlap][flow_bool[:,:overlap] !=0]) 
						BC_coor = np.mean(pred_field[:overlap,:][flow_bool[:overlap,:]!=0]) - BC_ups[idx_j]
					else:
						BC_ant_0 = np.mean(old_pred_field[:,:overlap][flow_bool[:,:overlap] !=0]) 
						BC_coor = np.mean(pred_field[:,-overlap:][flow_bool[:,-overlap:]!=0]) - BC_ant_0											
				else:
					BC_coor = np.mean(pred_field[:overlap,:][flow_bool[:overlap,:]!=0]) - BC_ups[idx_j]
					if idx_j != 0 and idx_j != n_x:
						BC_ant_0 = np.mean(old_pred_field[:,:overlap][flow_bool[:,:overlap] !=0]) 
						BC_coor_2 = np.mean(pred_field[:,-overlap:][flow_bool[:,-overlap:]!=0]) - BC_ant_0
					
						# ## Apply the lowest correction ... less prone to problems...
						# if abs(BC_coor_2) < abs(BC_coor):
						# 	BC_coor = BC_coor_2

				## Applying correction
				pred_field -= BC_coor

				## Storing upward average pressure
				BC_ups[idx_j] = np.mean(pred_field[-overlap:,:][flow_bool[-overlap:,:] !=0])
				
				## Value stored to be used in the last row depends on p_i
				if idx_i == n_y:
					BC_ups[idx_j] = np.mean(pred_field[-(shape-p_i):,:][flow_bool[-(shape-p_i):,:] !=0])
			
			
			## LAST row
			else:

				## Calculating correction to be applied

				## In the last column the correction needs to be from above to keep consistency (with BC_ups)
				if idx_j == n_x:
					BC_coor = np.mean(pred_field[-p_i-overlap:-p_i,:][flow_bool[-p_i-overlap:-p_i,:]!=0]) - BC_ups[idx_j]
				else:
									
					#up
					y_0 = -p_i - overlap
					y_f = -p_i
					n_up_non_nans = (flow_bool[y_0:y_f,:]!=0).sum()
					# right side
					x_0 = shape_x -shape - (n_x-1)*(shape-overlap)
					n_right_non_nans = (flow_bool[:, x_0:]!=0).sum()

					# Give preference to "up" or "right" correction???
					## Giving it to "up" because it is being done everywhere else
					## only switch method if in the overlap region more than 90% of the values are NANs
					
					if (n_up_non_nans)/128**2 > 0.9:
						if idx_j == 0:
							intersect_zone_limit = overlap - p_j
							BC_ant_0 = np.mean(old_pred_field[:, :intersect_zone_limit][flow_bool[:, :intersect_zone_limit] !=0]) 
							BC_coor = np.mean(pred_field[:, -intersect_zone_limit:][flow_bool[:, -intersect_zone_limit:]!=0]) - BC_ant_0
						else:
							BC_ant_0 = np.mean(old_pred_field[:,:overlap][flow_bool[:,:overlap] !=0]) 
							BC_coor = np.mean(pred_field[:,-overlap:][flow_bool[:,-overlap:]!=0]) - BC_ant_0								
					else:
						BC_coor = np.mean(pred_field[:-p_i,:][flow_bool[:-p_i,:]!=0]) - BC_ups[idx_j]

						if idx_j != 0:
							BC_ant_0 = np.mean(old_pred_field[:,:overlap][flow_bool[:,:overlap] !=0]) 
							BC_coor_2 = np.mean(pred_field[:,-overlap:][flow_bool[:,-overlap:]!=0]) - BC_ant_0
						
							# ## Apply the lowest correction ... less prone to problems...
							# if abs(BC_coor_2) < abs(BC_coor):
							# 	BC_coor = BC_coor_2
								
				## Applying the correction
				pred_field -= BC_coor
				
			old_pred_field = pred_field
			## Last reassembly step:
			## Assigning the block to the right location in the flow domain
			if [idx_i, idx_j] == [n_y + 1, 0]:
				result[-p_i:shape_y , 0:shape] = pred_field[-p_i:]
			elif idx_j == 0:
				result[(shape-overlap) * idx_i:(shape-overlap) * idx_i + shape, 0:shape] = pred_field
			## Last row
			elif idx_i == (n_y + 1):
				idx_j = n_x - idx_j
				# option 1 - thick penultimate row
				result[-p_i:, shape_x -shape - idx_j*(shape-overlap) :  shape_x- idx_j*(shape-overlap)] = pred_field[-p_i:]
				# option 2 - thick last row
				#result[-shape:, shape_x -shape - idx_j*(shape-overlap) :  shape_x- idx_j*(shape-overlap)] = pred_field

			else:
				idx_j = n_x - idx_j
				result[(shape-overlap) * idx_i:(shape-overlap) * idx_i + shape, shape_x -shape - idx_j*(shape-overlap) : shape_x- idx_j*(shape-overlap)] = pred_field
				
		t1 = time.time()
		if verbose_g:
			print( "Reassembly algorithm took:" + str(t1-t0) + " s")
		
		# Ensuring the constant pressure boundary condition to be ensured at the right-most cell center
		# instead of doing that at the boundary patch, brings bias to the domain, removing it below:
		result -= np.mean( 3* result[:,-1] - result[:,-2] )/3
		
		# result is the deltaP predicted by the model
		change_in_deltap = result - deltaP_prev_grid

		t0 = time.time()
		# Two passes sufficient for blending weight smoothing (3rd pass has diminishing returns)
		_tmp = np.empty_like(deltaU_change_grid)
		ndimage.uniform_filter(deltaU_change_grid, size=301, mode='constant', output=_tmp)
		ndimage.uniform_filter(_tmp, size=301, mode='constant', output=deltaU_change_grid)
		t1 = time.time()
		if verbose_g:
			print( "Applying blending filter:" + str(t1-t0) + " s")

		# weighted change_in_deltap
		# this will ignore changes in deltaP in regions where there is no change in deltaU
		change_in_deltap = change_in_deltap * deltaU_change_grid

		if apply_filter_g:
			t0 = time.time()
			# uniform_filter (box filter) is O(n) regardless of kernel size.
			# Three passes approximate a Gaussian by the central limit theorem.
			# size = 2*int(sigma*truncate+0.5)+1 matches Gaussian reach at truncate=3.0.
			# Reuses _tmp (same shape/dtype, no longer needed after blending multiply above).
			_sz = (2 * int(filter_tuple_g[0] * 3.0 + 0.5) + 1,
			       2 * int(filter_tuple_g[1] * 3.0 + 0.5) + 1)
			ndimage.uniform_filter(change_in_deltap, size=_sz, mode='constant', output=_tmp)
			ndimage.uniform_filter(_tmp, size=_sz, mode='constant', output=change_in_deltap)
			t1 = time.time()
			if verbose_g:
				print( "Applying filter took:" + str(t1-t0) + " s")

		# # Plotting the integrated pressure field
		## FOR DEBUGGING PURPOSES
		# fig, axs = plt.subplots(4,1, figsize=(30, 15))

		# no_flow_bool = grid[0,:,:,2] == 0
	
		# masked_arr = np.ma.array(change_in_deltap, mask=no_flow_bool)
		# cf = axs[0].imshow(masked_arr, interpolation='nearest', cmap='jet')#, vmax = vmax, vmin = vmin )
		# plt.colorbar(cf, ax=axs[0])

		# masked_arr = np.ma.array(grid[0,:,:,0], mask=no_flow_bool)
		# cf = axs[1].imshow(masked_arr, interpolation='nearest', cmap='jet')#, vmax = vmax, vmin = vmin )
		# plt.colorbar(cf, ax=axs[1])

		# masked_arr = np.ma.array(grid[0,:,:,1], mask=no_flow_bool)
		# cf = axs[2].imshow(masked_arr, interpolation='nearest', cmap='jet')#, vmax = vmax, vmin = vmin )
		# plt.colorbar(cf, ax=axs[2])

		# masked_arr = np.ma.array(deltaU_change_grid, mask=no_flow_bool)
		# cf = axs[3].imshow(masked_arr, interpolation='nearest', cmap='jet')#, vmax = vmax, vmin = vmin )
		# plt.colorbar(cf, ax=axs[3])

		# plt.savefig('fields.png')

		# CHANGED: use cached int32 index arrays instead of tuple(indices.T)
		change_in_deltap = change_in_deltap[indices_i, indices_j]

		t0 = time.time()
		# Interpolation into the orginal grid
		# Takes virtually no time because "vert" and "weigths" where already calculated on the init_func
		change_in_deltap = interpolate_fill_njit(change_in_deltap.ravel().astype(np.float32), vert_NPtoOF_i32, weights_NPtoOF_f32)
		#p_interp[np.isnan(p_interp)] = 0
		t1 = time.time()
		if verbose_g:
			print( "Final Interpolation took:" + str(t1-t0) + " s")

		#########################################################################
		#### The following code is a workaround to avoid the model to predict the pressure field near walls ####
		# Using the last time step pressure field for the near wall locations
		# Because we know that the model underperforms at the grid elements near walls
		# sdf_mesh = interpolate_fill(sdfunct[:,:,0], vert_NPtoOF, weights_NPtoOF)
		# p_interp[sdf_mesh < 0.05] = 0	

		# # Filling it with zeros means that the pressure previous time-step pressure field will be used
		# # IDW is used for extrapolation, thus this should not be necessary
		# p_interp[np.isnan(p_interp)] = np.zeros_like(p_interp[np.isnan(p_interp)])
		#################### This is not currently in use #########################

		p = deltaP_prev + change_in_deltap

		t1_py_func = time.time()
		if verbose_g:
			print( "The whole python function took : " + str(t1_py_func-t0_py_func) + " s")

		# CHANGED: vectorized split using np.split (faster than manual loop)
		p_rankwise = np.split(p, np.cumsum(len_rankwise)[:-1])
	else:
		p_rankwise = None
	
	p = comm.scatter(p_rankwise, root = 0)

	if rank ==0:
		print(memory())

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
