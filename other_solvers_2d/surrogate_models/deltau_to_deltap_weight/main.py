import time
import traceback
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np

import mpi4py
mpi4py.rc.initialize = True
mpi4py.rc.finalize = False
from mpi4py import MPI

import pickle as pk
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

from surrogate_models.deltau_to_deltap_weight.utils import *

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

def load_pca_and_NN(ipca_input_fn, ipca_output_fn, maxs_fn, PCA_std_vals_fn, weights_fn, var, model_arch, apply_filter, overlap_ratio, filter_tuple, verbose=True):
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
	global apply_filter_g, overlap_ratio_g, verbose_g, filter_tuple_g
	apply_filter_g = apply_filter
	overlap_ratio_g = overlap_ratio
	verbose_g = verbose
	filter_tuple_g = filter_tuple

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
	comp_input = pcainput.components_[:PC_input,:]
	pca_mean_input = pcainput.mean_

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
	model.load_weights(weights_fn)

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
		delta = 5e-3 

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
		block_size = 128

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

		deltaUx_interp = interpolate_fill(deltaUx_adim, vert_OFtoNP, weights_OFtoNP)
		deltaUy_interp = interpolate_fill(deltaUy_adim, vert_OFtoNP, weights_OFtoNP)
		deltaU_changed_interp = interpolate_fill(deltaU_changed, vert_OFtoNP, weights_OFtoNP)
		deltaP_prev_interp = interpolate_fill(deltaP_prev, vert_OFtoNP, weights_OFtoNP)

		t1 = time.time()
		if verbose_g:
			print( "1st interpolation took:" + str(t1-t0) + " s")

		t0 = time.time()
		grid = np.zeros(shape=(1, shape_y, shape_x, 3))

		# Rearrange interpolated 1D arrays into 2D arrays
		grid[0, :, :, 0:1][tuple(indices.T)] = deltaUx_interp.reshape(deltaUx_interp.shape[0], 1)
		grid[0, :, :, 1:2][tuple(indices.T)] = deltaUy_interp.reshape(deltaUy_interp.shape[0], 1)
		grid[0, :, :, 2:3] = sdfunct

		deltaP_prev_grid = np.zeros(shape=(shape_y, shape_x))
		deltaU_change_grid = np.zeros(shape=(shape_y, shape_x))

		deltaP_prev_grid[tuple(indices.T)] = deltaP_prev_interp.reshape(deltaP_prev_interp.shape[0])
		deltaU_change_grid[tuple(indices.T)] = deltaU_changed_interp.reshape(deltaU_changed_interp.shape[0])

		## Rescale input variables to [-1,1]
		grid[0,:,:,0:1] = grid[0,:,:,0:1] / max_abs_ux
		grid[0,:,:,1:2] = grid[0,:,:,1:2] / max_abs_uy
		grid[0,:,:,2:3] = grid[0,:,:,2:3] / max_abs_dist

		t1 = time.time()
		if verbose_g:
			print( "2nd interpolation took:" + str(t1-t0) + " s")

		# Setting any nan value to 0 to avoid issues
		grid[np.isnan(grid)] = 0

		x_list = []
		obst_list = []
		indices_list = []

		shape = 128
		overlap = int(overlap_ratio_g*shape)

		n_x = int(np.ceil((shape_x-shape)/(shape - overlap )) )
		n_y = int((shape_y-shape)/(shape - overlap ))

		t0 = time.time()

		# Sampling blocks of size [shape X shape] from the input fields (Ux, Uy and sdf)
		# In the indices_list, the indices corresponding to each sampled block is stored to enable domain reconstruction later.
		for i in range ( n_y + 2 ): #+1 b
			for j in range ( n_x +1 ):
				
				# going right to left
				x_0 = grid.shape[2] - j*shape + j*overlap - shape
				if j == n_x: x_0 = 0
				x_f = x_0 + shape

				y_0 = i*shape - i*overlap
				if i == n_y + 1: y_0 = grid.shape[1]-shape
				y_f = y_0 + shape

				x_list.append(grid[0:1, y_0:y_f, x_0:x_f, 0:3])

				indices_list.append([i, n_x - j])

		x_array = np.concatenate(x_list)

		t1 = time.time()
		if verbose_g:
			print( "Data extraction loop took:" + str(t1-t0) + " s")

		t0 = time.time()
		N = x_array.shape[0]
		features = x_array.shape[3]

		x_array_flat = x_array.reshape((N, x_array.shape[1]*x_array.shape[2], features ))

		input_flat = x_array_flat.reshape((x_array_flat.shape[0],-1))
		#input_transformed = pcainput.transform(input_flat)[:,:PC_input]
		input_transformed = np.dot(input_flat - pca_mean_input, comp_input.T)

		# Standardize input PCs
		x_input = (input_transformed - mean_in) / std_in

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

		# Getting the non-standerdized PCs
		res_concat = (res_concat * std_out) + mean_out

		res_flat_inv = np.dot(res_concat, comp_p) + pca_mean_p	
		res_concat = res_flat_inv.reshape((res_concat.shape[0], shape, shape, 1))
		t1 = time.time()
		if verbose_g:
			print( "PCA inverse transform : " + str(t1-t0) + " s")

		# Redimensionalizing the predicted pressure field
		res_concat = res_concat * max_abs_p * pow(U_max_norm, 2.0)

		for i, x in enumerate(x_list):
			if (x[0,:,:,0] < 1e-5).all() and (x[0,:,:,1] < 1e-5).all():
				res_concat[i] = np.zeros((shape, shape, 1))

		# The boundary condition is a fixed pressure of 0 at the output
		Ref_BC = 0 

		## Domain reassembly method
		result = np.empty(shape=(shape_y, shape_x))

		## Array to store the average pressure in the overlap region with the next down block
		BC_ups = np.zeros(n_x+1)

		# i index where the lower blocks are located
		p_i = shape_y - ( (shape-overlap) * n_y + shape )
		
		# j index where the left-most blocks are located
		p_j = shape_x - ( (shape - overlap) * n_x + shape )

		t0 = time.time()

		for i in range(len(x_list)):

			idx_i, idx_j = indices_list[i]
			flow_bool = x_array[i,:,:,2]
			pred_field = res_concat[i,:,:,0]
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
		# Apply Gaussian filter to smooth the blending function
		deltaU_change_grid = ndimage.gaussian_filter(deltaU_change_grid, sigma=(50,50), order=0)
		#deltaU_change_grid = deltaU_change_grid/deltaU_change_grid.max()
		t1 = time.time()
		if verbose_g:
			print( "Applying Gaussian filter to blending function:" + str(t1-t0) + " s")

		# weighted change_in_deltap
		# this will ignore changes in deltaP in regions where there is no change in deltaU
		change_in_deltap = change_in_deltap * deltaU_change_grid

		if apply_filter_g:
			t0 = time.time()
			# Apply Gaussian filter to correct the attained change in deltap field (and remove artifacts) (OPTIONAL)
			change_in_deltap = ndimage.gaussian_filter(change_in_deltap, sigma=filter_tuple_g, order=0)
			t1 = time.time()
			if verbose_g:
				print( "Applying Gaussian filter took:" + str(t1-t0) + " s")

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

		init = 0
		# Dividing p into a list of n elements (consistent with the OF domain decomposition)
		# This is necessary to enable parallelization in OF
		p_rankwise = [] 
		for length in len_rankwise:
		    end = init + length
		    p_rankwise.append(p[init:end,...])
		    init += length
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
