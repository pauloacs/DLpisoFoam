import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

from numba import njit
import os
import shutil
import numpy as np

import pickle as pk
import matplotlib.pyplot as plt
import tensorly as tl

import scipy.ndimage as ndimage

from . import assembly
from .utils import io_operations as utils_io
from .utils import data_processing
from .utils import domain_geometry
from .utils import sampling
from .utils import visualization

class Evaluation():
	def __init__(
		self,
		delta: float,
		block_size: int,
		overlap: float,
		dataset_path: str,
		model_path: str,
		standardization_method: str,
		ranks):
		"""
		Initialize Evaluation class. 

		Args:
			delta (float): The value of delta.
			block_size (int): The shape value.
			overlap (float): The overlap value.
			dataset_path (str): The path to the dataset.
			model_path (str): The path to the model.
			standardization_method (str): The standardization method.

		Attributes:
			delta (float): The value of delta.
			block_size (int): The shape value.
			overlap (float): The overlap value.
			dataset_path (str): The path to the dataset.
			model_path (str): The path to the model.
			standardization_method (str): The standardization method.
			max_abs_delta_Ux (float): The maximum absolute value of delta Ux.
			max_abs_delta_Uy (float): The maximum absolute value of delta Uy.
			max_abs_dist (float): The maximum absolute value of dist.
			max_abs_delta_p (float): The maximum absolute value of delta p.
			model (tf.keras.Model): The loaded model.
		"""
		self.delta = delta
		self.shape = block_size
		self.overlap = overlap
		self.dataset_path = dataset_path
		self.standardization_method = standardization_method

		maxs = np.loadtxt('maxs')

		(self.max_abs_delta_Ux,
			self.max_abs_delta_Uy,
			self.max_abs_delta_Uz, 
			self.max_abs_dist,
			self.max_abs_delta_p) = maxs

		#### loading the model #######
		if 'MLP_attention_biased' in model_path:
			from .train import BiasedAttention
			self.model = tf.keras.models.load_model(model_path, custom_objects={'BiasedAttention': BiasedAttention})
		elif 'fno3d' in model_path.lower():
			from .neural_networks import SpectralConv3D, FNOBlock3D
			self.model = tf.keras.models.load_model(model_path, custom_objects={'FNOBlock3D': FNOBlock3D})
			#self.model = tf.keras.models.load_model(model_path, custom_objects={'SpectralConv3D': SpectralConv3D})
		else:
			self.model = tf.keras.models.load_model(model_path)
		print(self.model.summary())
		
		### loading the pca matrices for transformations ###
		with open('tucker_factors.pkl', 'rb') as f:
			factors = pk.load(f)
			self.input_factors = factors['input_factors']
			self.output_factors = factors['output_factors']

		self.input_shape = ranks * ranks * ranks * 4 
		self.output_shape = ranks * ranks * ranks
		
	def computeOnlyOnce(self, sim):
		"""
		Performs interpolation from the OF grid (corresponding to the mesh cell centers),
		saves the intepolation vertices and weights and computes the signed distance function (sdf).

		Args:
			sim (int): Simulation number.
		"""
		time = 0

		boundaries = utils_io.read_boundaries(sim, self.dataset_path)

		data, limits = utils_io.read_cells_and_limits(self.dataset_path, sim, 0, 1)
		data_t0 = data[0, ...]

		######### -------------------- Assuming constant mesh, the following can be done out of the for cycle ------------------------------- ##########

		X0, Y0, Z0 = data_processing.create_uniform_grid(limits, self.delta)

		xyz0 = np.concatenate((np.expand_dims(X0, axis=1), np.expand_dims(Y0, axis=1), np.expand_dims(Z0, axis=1)), axis=-1)
		
		points = data_t0[:, 4:7] #coordinates
		self.vert, self.weights = data_processing.interp_weights(points, xyz0, interp_method='IDW') #takes ~100% of the interpolation cost and it's only done once for each different mesh/simulation case

		domain_bool, sdf = domain_geometry.domain_dist(boundaries, xyz0, self.delta)

		#div defines the sliding window vertical and horizontal displacements
		div = 1 

		self.grid_shape_z = int(round((limits['z_max']-limits['z_min'])/self.delta))
		self.grid_shape_y = int(round((limits['y_max']-limits['y_min'])/self.delta)) #+1
		self.grid_shape_x = int(round((limits['x_max']-limits['x_min'])/self.delta)) #+1

		#arrange data in array: #this can be put outside the j loop if the mesh is constant 

		x0 = np.min(X0)
		y0 = np.min(Y0)
		z0 = np.min(Z0)

		dx = self.delta
		dy = self.delta
		dz = self.delta

		indices= np.zeros((X0.shape[0],3))
		obst_bool = np.zeros((self.grid_shape_z, self.grid_shape_y, self.grid_shape_x, 1))
		self.sdfunct = obst_bool.copy()

		p = data_t0[:,2:3]
		p_interp = data_processing.interpolate_fill(p, self.vert, self.weights) 

		for (step, x_y_z) in enumerate(xyz0):  
			if domain_bool[step] * (~np.isnan(p_interp[step])) :
				ii = int(round((x_y_z[...,2] - z0) / dz))
				jj = int(round((x_y_z[...,1] - y0) / dy))
				kk = int(round((x_y_z[...,0] - x0) / dx))

				indices[step,0] = ii
				indices[step,1] = jj
				indices[step,2] = kk

				self.sdfunct[ii, jj, kk, :] = sdf[step]
				obst_bool[ii, jj, kk, :]  = int(1)

		self.indices = indices.astype(int)

	def timeStep(
		self,
		sim: int,
		time: int,
		plot_intermediate_fields: bool,
		save_plots: bool,
		show_plots: bool,
		apply_filter: bool,
		apply_deltaU_change_wgt: bool,
		flatten_data: bool):
		"""
		Performs a time step in the simulation.

		Args:
			sim (int): The simulation number.
			time (int): The time step number.
			plot_intermediate_fields (bool): Whether to plot intermediate fields during the time step.
			save_plots (bool): Whether to save the plots.
			show_plots (bool): Whether to display the plots.
			apply_filter (bool): Whether to apply the filter.

		Returns:
			None
		"""

		data, _ = utils_io.read_cells_and_limits(self.dataset_path, sim, time, time+1)
		data = data[0, ...]  #shape: (n_cells, n_features)

		Ux =  data[:,0:1]
		Uy =  data[:,1:2]
		Uz = data[:,2:3]

		delta_U = data[:,7:10]
		delta_Ux = delta_U[...,0:1]
		delta_Uy = delta_U[...,1:2]
		delta_Uz = delta_U[...,2:3]

		delta_U_prev = data[:, 11:14]
		delta_p_prev = data[:,14:15]

		# check where the deltaU has changed in the last time step
		#deltaU_changed = np.abs(delta_U - delta_U_prev).sum(axis=-1)
		#deltaU_changed = deltaU_changed / deltaU_changed.max()

		# For accuracy accessment
		delta_p = data[:,10:11] #values
		p = data[:,3:4] #values

		U_max_norm = np.max(np.sqrt(np.square(Ux) + np.square(Uy) + np.square(Uz)))
		deltaU_max_norm = np.max(np.sqrt(np.square(delta_Ux) + np.square(delta_Uy), np.square(delta_Uz)))

		# Ignore time steps with minimal changes ...
		# there is not point in computing error metrics for these
		# it would exagerate the delta_p errors and give ~0% errors in p
		threshold = 1e-4
		irrelevant_ts = (deltaU_max_norm/U_max_norm) < threshold

		if irrelevant_ts:
			print(f"\n\n Irrelevant time step, U_norm changed {(deltaU_max_norm/U_max_norm)*100:.2f} [%] (criteria: < {threshold*100:.2f} [%]).\n Skipping Time step.")
			return 0

		delta_p_adim = delta_p / pow(U_max_norm,2.0) 
		delta_Ux_adim = delta_Ux/U_max_norm
		delta_Uy_adim = delta_Uy/U_max_norm
		delta_Uz_adim = delta_Uz/U_max_norm

		Ux_adim = Ux/U_max_norm
		Uy_adim = Uy/U_max_norm
		Uz_adim = Uz/U_max_norm
		p_adim = p / pow(U_max_norm,2.0)
		
		# Interpolate the data to the grid
		Ux_interp = data_processing.interpolate_fill(Ux_adim, self.vert, self.weights)
		Uy_interp = data_processing.interpolate_fill(Uy_adim, self.vert, self.weights)
		Uz_interp = data_processing.interpolate_fill(Uz_adim, self.vert, self.weights)
		p_interp = data_processing.interpolate_fill(p_adim, self.vert, self.weights)


		delta_p_interp = data_processing.interpolate_fill(delta_p_adim, self.vert, self.weights)
		delta_Ux_interp = data_processing.interpolate_fill(delta_Ux_adim, self.vert, self.weights)
		delta_Uy_interp = data_processing.interpolate_fill(delta_Uy_adim, self.vert, self.weights)
		delta_Uz_interp = data_processing.interpolate_fill(delta_Uz_adim, self.vert, self.weights)
		p_interp = data_processing.interpolate_fill(p, self.vert, self.weights)

		# weighting 
		#deltaU_changed_interp = data_processing.interpolate_fill(deltaU_changed, self.vert, self.weights)
		#delta_p_prev_interp = data_processing.interpolate_fill(delta_p_prev, self.vert, self.weights)

		grid = np.zeros(shape=(1, self.grid_shape_z, self.grid_shape_y, self.grid_shape_x, 9))
		filter_tuple = (2,2,2)
		grid[0,:,:,:,0:1][tuple(self.indices.T)] = delta_Ux_interp.reshape(delta_Ux_interp.shape[0], 1)
		#grid[0,:,:,:,0] = ndimage.gaussian_filter(grid[0,:,:,:,0], sigma=filter_tuple, order=0)

		grid[0,:,:,:,1:2][tuple(self.indices.T)] = delta_Uy_interp.reshape(delta_Uy_interp.shape[0], 1)
		#grid[0,:,:,:,1] = ndimage.gaussian_filter(grid[0,:,:,:,1], sigma=filter_tuple, order=0)

		grid[0,:,:,:,2:3][tuple(self.indices.T)] = delta_Uz_interp.reshape(delta_Uz_interp.shape[0], 1)
		#grid[0,:,:,:,2] = ndimage.gaussian_filter(grid[0,:,:,:,2], sigma=filter_tuple, order=0)
		
		grid[0,:,:,:,3:4] = self.sdfunct
		grid[0,:,:,:,4:5][tuple(self.indices.T)] = delta_p_interp.reshape(delta_p_interp.shape[0], 1)
		grid[0,:,:,:,5:6][tuple(self.indices.T)] = p_interp.reshape(p_interp.shape[0], 1)

		#### REMOVE AFTER PLOTS

		grid[0,:,:,:,6:7][tuple(self.indices.T)] = Ux_interp.reshape(delta_p_interp.shape[0], 1)
		grid[0,:,:,:,7:8][tuple(self.indices.T)] = Uz_interp.reshape(delta_p_interp.shape[0], 1)
		grid[0,:,:,:,8:9][tuple(self.indices.T)] = Uy_interp.reshape(p_interp.shape[0], 1)

		grid[np.isnan(grid)] = 0 #set any nan value to 0

		## Rescale all the variables to [-1,1]
		grid[0,:,:,:,0:1] = grid[0,:,:,:,0:1]/self.max_abs_delta_Ux
		grid[0,:,:,:,1:2] = grid[0,:,:,:,1:2]/self.max_abs_delta_Uy
		grid[0,:,:,:,2:3] = grid[0,:,:,:,2:3]/self.max_abs_delta_Uz
		grid[0,:,:,:,3:4] = grid[0,:,:,:,3:4]/self.max_abs_dist
		grid[0,:,:,:,4:5] = grid[0,:,:,:,4:5]/self.max_abs_delta_p

		#if save_plots:
			#visualization.plot_inputs_slices(grid[0,...,0], grid[0,...,1], grid[0,...,2], \
		#					grid[0,...,3], grid[0,...,4], slices_indices=[5, 10, 20])

		# saving for weighting procedure
		#deltaU_change_grid = np.zeros(shape=(self.grid_shape_z, self.grid_shape_y, self.grid_shape_x))
		#deltaU_change_grid[tuple(self.indices.T)] = deltaU_changed_interp.reshape(deltaU_changed_interp.shape[0])
		#deltaP_prev_grid = np.zeros(shape=(self.grid_shape_z, self.grid_shape_y, self.grid_shape_x))
		#deltaP_prev_grid[tuple(self.indices.T)] = delta_p_prev_interp.reshape(delta_p_prev_interp.shape[0])
		
		# Initialize to None since the weighting feature is currently disabled
		deltaU_change_grid = None
		deltaP_prev_grid = None

		## Block extraction
		x_list = []
		obst_list = []
		y_list = []
		indices_list = []

		overlap = self.overlap
		shape = self.shape

		n_x = int(np.ceil((grid.shape[3]-shape)/(shape - overlap )) ) + 1
		n_y = int(np.ceil((grid.shape[2]-shape)/(shape - overlap )) ) + 1
		n_z = int(np.ceil((grid.shape[1]-shape)/(shape - overlap )) ) + 1

		for i in range (n_z):
			z_0 = i*shape - i*overlap
			if i == n_z - 1:
				z_0 = grid.shape[1]-shape
			z_f = z_0 + shape
			for j in range (n_y):
				y_0 = j*shape - j*overlap
				if j == n_y - 1:
					y_0 = grid.shape[2]-shape
				y_f = y_0 + shape
				for k in range(n_x):
					# going right to left
					x_0 = grid.shape[3] - k*shape + k*overlap - shape
					if k == n_x - 1:
						x_0 = 0
					x_f = x_0 + shape

					#DEBUGGING print
					#print(f"{(z_0,z_f, y_0,y_f, x_0,x_f)}")
					#print(f"indices: {(i, j, n_x -1 - k)}")

					x_list.append(grid[0:1, z_0:z_f, y_0:y_f, x_0:x_f, 0:4])
					y_list.append(grid[0:1, z_0:z_f, y_0:y_f, x_0:x_f, 4:5])

					indices_list.append([i, j, n_x -1 - k])

		# for i, block in enumerate(x_list[:100]):
		# 	fig, axs = plt.subplots(1, 5, figsize=(20, 4))
		# 	for j in range(4):
		# 		im = axs[j].imshow(block[0, 8, :, :, j], cmap='viridis')
		# 		axs[j].set_title(f'Feature {j}')
		# 		axs[j].axis('off')
		# 		plt.colorbar(im, ax=axs[j], orientation='vertical')
		# 	axs[4].imshow(y_list[i][0, 8, :, :, 0], cmap='viridis')
		# 	plt.tight_layout()
		# 	plt.savefig(f'blocks_inputs/blocks_inputs_{i}.png')
		# 	plt.close(fig)

		self.x_array = np.concatenate(x_list)
		self.y_array = np.concatenate(y_list)

		y_array = self.y_array
		N = self.x_array.shape[0]
		features = self.x_array.shape[4]
		
		for step in range(y_array.shape[0]):
			y_array[step,...,0][self.x_array[step,...,3] != 0] -= np.mean(y_array[step,...,0][self.x_array[step,...,3] != 0])

		# Apply Tucker decomposition to transform input and output tensors
		input_core = tl.tenalg.multi_mode_dot(self.x_array, self.input_factors[1:], modes=[1, 2, 3, 4], transpose=True)
		output_core = tl.tenalg.multi_mode_dot(y_array[...,0], self.output_factors[1:], modes=[1, 2, 3], transpose=True)

		if flatten_data:
			input_core_flat = input_core.reshape(N, -1)
			output_core_flat = output_core.reshape(N, -1)
		else: 
			input_core_flat = input_core
			output_core_flat = output_core

		if self.standardization_method == 'std':
			## Option 1: Standardization
			data = np.load('mean_std.npz')
			mean_in_loaded = data['mean_in']
			std_in_loaded = data['std_in']
			mean_out_loaded = data['mean_out']
			std_out_loaded = data['std_out']
			x_input = (input_core_flat - mean_in_loaded) / std_in_loaded
		elif self.standardization_method == 'min_max':
			## Option 2: Min-max scaling
			data = np.load('min_max_values.npz')
			min_in_loaded = data['min_in']
			max_in_loaded = data['max_in']
			min_out_loaded = data['min_out']
			max_out_loaded = data['max_out']
			x_input = (input_core_flat - min_in_loaded) / (max_in_loaded - min_in_loaded)
		elif self.standardization_method == 'max_abs':
			## Option 3: Old method
			x_input = input_core_flat / self.max_abs_input_PCA
		else:
			raise ValueError("Standardization method not valid")

		res_concat = np.array(self.model(np.array(x_input)))

		if self.standardization_method == 'std':
			res_concat = (res_concat * std_out_loaded) + mean_out_loaded
		elif self.standardization_method == 'min_max':
			res_concat = res_concat * (max_out_loaded - min_out_loaded) + min_out_loaded
		elif self.standardization_method == 'max_abs':
			res_concat *= self.max_abs_output_PCA
		else:
			raise ValueError("Standardization method not valid")

		# Reshape res_concat to match the original shape of y_array
		res_concat = res_concat.reshape(output_core.shape)
		y_array = output_core_flat.reshape(output_core.shape)

		# Perform inverse transformation using Tucker factors
		res_concat = tl.tenalg.multi_mode_dot(res_concat, self.output_factors[1:], modes=[1, 2, 3], transpose=False)
		y_concat = tl.tenalg.multi_mode_dot(y_array, self.output_factors[1:], modes=[1, 2, 3], transpose=False)

		res_concat = res_concat.reshape(self.y_array.shape)
		## Dimensionalize pressure field - There is no need to dimensionalize the pressure field here
		## As we can compare it to the reference non-dimensionalized field
		## This only needs to be done when calling the SM in the CFD solver
		res_concat = res_concat  * self.max_abs_delta_p * pow(U_max_norm,2.0)

		## Here compute the error only based on the blocks pressure fields - before the assembly
		flow_bool = self.x_array[...,3:4] != 0

		pred_minus_true_block, pred_minus_true_squared_block = sampling.compute_in_block_error(res_concat, self.y_array * self.max_abs_delta_p * pow(U_max_norm,2.0), flow_bool)
		self.pred_minus_true_block.append(pred_minus_true_block)
		self.pred_minus_true_squared_block.append(pred_minus_true_squared_block)
		
		#visualization.plot_random_blocks(res_concat, y_array, self.x_array, sim, time, save_plots)

		#### This gives worse results... #####
		# Ignore blocks with near zero delta_U
		# Assign deltap = 0
		# for i, x in enumerate(x_list):
		# 	if (x[0,:,:,0] < 1e-5).all() and (x[0,:,:,1] < 1e-5).all():
		# 		res_concat[i] = np.zeros((shape, shape, 1))
		#### This gives worse results... #####
		
		# the boundary condition is a fixed pressure of 0 at the output
		self.Ref_BC = 0 

		# # performing the assembly process
		#apply_deltaU_change_wgt = False
		# deltap_res, change_in_deltap = self.assemble_prediction(
		# 	res_concat[..., 0],
		# 	indices_list,
		# 	n_x,
		# 	n_y,
		# 	n_z,
		# 	apply_filter,
		# 	grid.shape[3],
		# 	grid.shape[2],
		# 	grid.shape[1],
		# 	deltaU_change_grid,
		# 	deltaP_prev_grid,
		# 	apply_deltaU_change_wgt
		# )

		# Call the assemble_prediction from utils for comparison or debugging
		deltap_res, change_in_deltap = assembly.assemble_prediction(
			res_concat[..., 0],
			indices_list,
			n_x,
			n_y,
			n_z,
			self.overlap,
			self.shape,
			self.Ref_BC,
			self.x_array,
			apply_filter,
			grid.shape[3],
			grid.shape[2],
			grid.shape[1],
			deltaU_change_grid=None,
			deltaP_prev_grid=None,
			apply_deltaU_change_wgt=apply_deltaU_change_wgt,
		)

		# The next line can be used to evaluate the assembly algorithm
		#y_array = y_array  * self.max_abs_delta_p * pow(U_max_norm,2.0)
		#deltap_res, _ = self.assemble_prediction(y_array[...,0], indices_list, n_x, n_y, n_z, apply_filter,
		#							grid.shape[3], grid.shape[2], grid.shape[1], deltaU_change_grid, deltaP_prev_grid, apply_deltaU_change_wgt)

		# use field_deltap = deltap_test_res to test the assembly algorith -> it should be almost perfect in that case
		if not apply_deltaU_change_wgt:
			# option 1: use pure deltap
			field_deltap = deltap_res
		else:
			# option 2: use the change in deltap
			field_deltap = deltaP_prev_grid + change_in_deltap

		cfd_results = grid[0,:,:,:,4] * self.max_abs_delta_p * pow(U_max_norm,2.0)
		no_flow_bool = grid[0,:,:,:,3] == 0

		if save_plots:
			visualization.plot_delta_p_comparison(
				cfd_results,
				field_deltap,
				no_flow_bool,
				slices_indices=[5, 20, 40, 60, 80, 95],
				fig_path=f'plots/sim{sim}/deltap_pred_t{time}.png')
			
			visualization.plot_delta_p_comparison_slices(
				cfd_results,
				field_deltap,
				no_flow_bool,
				slices_indices=[5, 20, 40, 60, 80, 95],
				fig_path=f'plots/sim{sim}/deltap_pred_t{time}_slices.png')
			
			# if time == 9:
			# 	visualization.plot_cfd_results_3d_helper(cfd_results[:,:, 20:150], no_flow_bool[:,:, 20:150], slices_indices=[0, 8, 16, 24], fig_path=f'plots/sim{sim}/deltap_p{time}_true.png')
			# 	visualization.plot_cfd_results_3d_helper(grid[0,:,:,:,0][:,:, 20:150], no_flow_bool[:,:, 20:150], slices_indices=[0, 8, 16, 24], fig_path=f'plots/sim{sim}/deltaux{time}_true.png')
			# 	visualization.plot_cfd_results_3d_helper(grid[0,:,:,:,1][:,:, 20:150], no_flow_bool[:,:, 20:150], slices_indices=[0, 8, 16, 24], fig_path=f'plots/sim{sim}/deltauy{time}_true.png')
			# 	visualization.plot_cfd_results_3d_helper(grid[0,:,:,:,2][:,:, 20:150], no_flow_bool[:,:, 20:150], slices_indices=[0, 8, 16, 24], fig_path=f'plots/sim{sim}/deltauz{time}_true.png')
			# 	visualization.plot_cfd_results_3d_helper(grid[0,:,:,:,3][:,:, 20:150], no_flow_bool[:,:, 20:150], slices_indices=[0, 8, 16, 24], fig_path=f'plots/sim{sim}/sdf{time}_true.png')


			# 	visualization.plot_cfd_results_3d_helper(grid[0,:,:,:,5][:,:, 20:150], no_flow_bool[:,:, 20:150], slices_indices=[0, 8, 16, 24], fig_path=f'plots/sim{sim}/p{time}_true.png')
			# 	visualization.plot_cfd_results_3d_helper(grid[0,:,:,:,6][:,:, 20:150], no_flow_bool[:,:, 20:150], slices_indices=[0, 8, 16, 24], fig_path=f'plots/sim{sim}/ux{time}_true.png')
			# 	visualization.plot_cfd_results_3d_helper(grid[0,:,:,:,7][:,:, 20:150], no_flow_bool[:,:, 20:150], slices_indices=[0, 8, 16, 24], fig_path=f'plots/sim{sim}/uy{time}_true.png')
			# 	visualization.plot_cfd_results_3d_helper(grid[0,:,:,:,8][:,:, 20:150], no_flow_bool[:,:, 20:150], slices_indices=[0, 8, 16, 24], fig_path=f'plots/sim{sim}/uz{time}_true.png')


			# 	visualization.plot_cfd_results_3d_helper(cfd_results[0:16,0:16, 100:116], no_flow_bool[0:16,0:16, 100:116], slices_indices=[0,15], fig_path=f'plots/sim{sim}/block_deltap_t{time}_true.png', alpha_boundary=0.8)
			# 	visualization.plot_cfd_results_3d_helper(grid[0,:,:,:,0][0:16,0:16, 100:116], no_flow_bool[0:16,0:16, 100:116], slices_indices=[0, 15], fig_path=f'plots/sim{sim}/block_ux{time}_true.png', alpha_boundary=0.8)
			# 	visualization.plot_cfd_results_3d_helper(grid[0,:,:,:,1][0:16,0:16, 100:116], no_flow_bool[0:16,0:16, 100:116], slices_indices=[0, 15], fig_path=f'plots/sim{sim}/block_uy{time}_true.png', alpha_boundary=0.8)
			# 	visualization.plot_cfd_results_3d_helper(grid[0,:,:,:,2][0:16,0:16, 100:116], no_flow_bool[0:16,0:16, 100:116], slices_indices=[0, 15], fig_path=f'plots/sim{sim}/block_uz{time}_true.png', alpha_boundary=0.8)
			# 	visualization.plot_cfd_results_3d_helper(grid[0,:,:,:,3][0:16,0:16, 100:116], no_flow_bool[0:16,0:16, 100:116], slices_indices=[0, 15], fig_path=f'plots/sim{sim}/block_sdf{time}_true.png', alpha_boundary=0.8)


		#visualization.simple_delta_p_slices_plot()
		
		# actual pressure fields

		# Infering p_t-1 from ref p and delta_p
		## grid[...,4] is p without being normalized to [0,1]
		## grid[...,3] was normalized ...

		p_CFD = grid[0,:,:,:,5]
		p_prev = p_CFD - cfd_results
		p_pred = p_prev + field_deltap

		# if save_plots or show_plots:
#             # Plotting the integrated pressure field
		# 	fig, axs = plt.subplots(3,1, figsize=(30, 15))

		# 	masked_arr = np.ma.array(p_pred, mask=no_flow_bool)
		# 	axs[0].set_title(r'Predicted pressure $p_{t-1} + delta_p$', fontsize = 15)
		# 	cf = axs[0].imshow(masked_arr, interpolation='nearest', cmap='viridis')#, vmax = vmax, vmin = vmin )
		# 	plt.colorbar(cf, ax=axs[0])

		# 	masked_arr = np.ma.array(p_CFD, mask=no_flow_bool)
		# 	axs[1].set_title('Pressure (CFD)', fontsize = 15)
		# 	cf = axs[1].imshow(masked_arr, interpolation='nearest', cmap='viridis')#, vmax = vmax, vmin = vmin)
		# 	plt.colorbar(cf, ax=axs[1])

		# 	masked_arr = np.ma.array( np.abs(( p_pred - p_CFD )/(np.max(p_CFD) -np.min(p_CFD))*100) , mask=no_flow_bool)

		# 	axs[2].set_title('error in %', fontsize = 15)
		# 	cf = axs[2].imshow(masked_arr, interpolation='nearest', cmap='viridis')#, vmax = 2, vmin=0 )
		# 	plt.colorbar(cf, ax=axs[2])

		# if show_plots:
		# 	plt.show()

		# if save_plots:
		# 	plt.savefig(f'plots/sim{sim}/p_pred_t{time}.png')

		# plt.close()

		# if show_plots or save_plots:
		# 	# # Plotting the input fields - for debugging purposes
		# 	fig, axs = plt.subplots(3,1, figsize=(65, 15))

		# 	masked_arr = np.ma.array(grid[0,:,:,:,0], mask=no_flow_bool)
		# 	axs[0].set_title(r'\Delta U_x', fontsize=15)
		# 	cf = axs[0].imshow(masked_arr, interpolation='nearest', cmap='viridis')#, vmax = vmax, vmin = vmin )
		# 	plt.colorbar(cf, ax=axs[0])

		# 	masked_arr = np.ma.array(grid[0,:,:,:,1], mask=no_flow_bool)
		# 	axs[1].set_title(r'\Delta U_y', fontsize=15)
		# 	cf = axs[1].imshow(masked_arr, interpolation='nearest', cmap='viridis')#, vmax = vmax, vmin = vmin)
		# 	plt.colorbar(cf, ax=axs[1])

		# 	masked_arr = np.ma.array(grid[0,:,:,:,2] , mask=no_flow_bool)
		# 	axs[2].set_title(r'\Delta U_z', fontsize=15)
		# 	cf = axs[2].imshow(masked_arr, interpolation='nearest', cmap='viridis') #, vmax = 10, vmin=0 )
		# 	plt.colorbar(cf, ax=axs[2])

		# 	masked_arr = np.ma.array(grid[0,:,:,:,3] , mask=no_flow_bool)
		# 	axs[3].set_title(r'\Delta \rho', fontsize=15)
		# 	cf = axs[2].imshow(masked_arr, interpolation='nearest', cmap='viridis') #, vmax = 10, vmin=0 )
		# 	plt.colorbar(cf, ax=axs[2])


		# if save_plots:
		# 	plt.savefig(f'plots/sim{sim}/inputs-t{time}.png')
		
		############## ------------------//------------------##############################
		
		true_masked = cfd_results[~no_flow_bool]
		pred_masked = field_deltap[~no_flow_bool]

		# Calculate norm based on reference data a predicted data
		norm_true = np.max(true_masked) - np.min(true_masked)
		norm_pred = np.max(pred_masked) - np.min(pred_masked)

		norm = norm_true

		mask_nan = ~np.isnan( pred_masked  - true_masked )

		BIAS_norm = np.mean( (pred_masked  - true_masked )[mask_nan] )/norm * 100
		RMSE_norm = np.sqrt(np.mean( ( pred_masked  - true_masked )[mask_nan]**2 ))/norm * 100
		STDE_norm = np.sqrt( (RMSE_norm**2 - BIAS_norm**2) )
		
		print(f"""
		norm_true = {norm_true};
		norm_pred = {norm_pred};

		** Error in delta_p **

			normVal  = {norm} Pa
			biasNorm = {BIAS_norm:.3f}%
			stdeNorm = {STDE_norm:.3f}%
			rmseNorm = {RMSE_norm:.3f}%
		""", flush = True)

		self.pred_minus_true.append( np.mean( (pred_masked  - true_masked )[mask_nan] )/norm )
		self.pred_minus_true_squared.append( np.mean( (pred_masked  - true_masked )[mask_nan]**2 )/norm**2 )
		
		## Error in crude deltaP - withouth weighting
		pred_masked = deltap_res[~no_flow_bool]

		norm_pred = np.max(pred_masked) - np.min(pred_masked)
		norm = norm_true #max(norm_true, norm_pred)

		BIAS_norm = np.mean( (pred_masked  - true_masked )[mask_nan] )/norm * 100
		RMSE_norm = np.sqrt(np.mean( ( pred_masked  - true_masked )[mask_nan]**2 ))/norm * 100
		STDE_norm = np.sqrt( (RMSE_norm**2 - BIAS_norm**2) )

		if apply_deltaU_change_wgt:
			print(f"""
			** Error in delta_p - no weighting **

				normVal  = {norm} Pa
				biasNorm = {BIAS_norm:.3f}%
				stdeNorm = {STDE_norm:.3f}%
				rmseNorm = {RMSE_norm:.3f}%
			""", flush = True)

		self.pred_minus_true_deltap_crude.append( np.mean( (pred_masked  - true_masked )[mask_nan] )/norm )
		self.pred_minus_true_squared_deltap_crude.append( np.mean( (pred_masked  - true_masked )[mask_nan]**2 )/norm**2 )

		## Error in p

		true_masked = p_CFD[~no_flow_bool]
		pred_masked = p_pred[~no_flow_bool]

		norm_true = np.max(true_masked) - np.min(true_masked)
		norm_pred = np.max(pred_masked) - np.min(pred_masked)
		norm = norm_true #max(norm_true, norm_pred)

		mask_nan = ~np.isnan( pred_masked  - true_masked )

		BIAS_norm = np.mean( (pred_masked  - true_masked )[mask_nan] )/norm * 100
		RMSE_norm = np.sqrt(np.mean( ( pred_masked  - true_masked )[mask_nan]**2 ))/norm * 100
		STDE_norm = np.sqrt( (RMSE_norm**2 - BIAS_norm**2) )

		print(f"""
		** Error in p **

			normVal  = {norm} Pa
			biasNorm = {BIAS_norm:.5f}%
			stdeNorm = {STDE_norm:.5f}%
			rmseNorm = {RMSE_norm:.5f}%
		""", flush = True)

		self.pred_minus_true_p.append( np.mean( (pred_masked  - true_masked )[mask_nan] )/norm )
		self.pred_minus_true_squared_p.append( np.mean( (pred_masked  - true_masked )[mask_nan]**2 )/norm**2 )

		return 0


def call_SM_main(delta, model_name, block_size, overlap_ratio, dataset_path, \
					plot_intermediate_fields, standardization_method, save_plots, show_plots, apply_filter, create_GIF, \
					first_sim, last_sim, first_t, last_t, ranks):

	if 'MLP'.lower() in model_name.lower():
		flatten_data = True
	else:
		flatten_data = False

	apply_deltaU_change_wgt = False

	if save_plots:
		path = 'plots/'
		if os.path.exists(path):
			shutil.rmtree(path)
		os.makedirs(path)

	overlap = int(overlap_ratio*block_size)
	
	Eval = Evaluation(delta, block_size, overlap, dataset_path, model_name, standardization_method, ranks)

	Eval.pred_minus_true_block = []
	Eval.pred_minus_true_squared_block = []

	Eval.pred_minus_true = []
	Eval.pred_minus_true_squared = []

	Eval.pred_minus_true_deltap_crude = []
	Eval.pred_minus_true_squared_deltap_crude = []

	Eval.pred_minus_true_p = []
	Eval.pred_minus_true_squared_p = []

	for sim in range(first_sim, last_sim + 1):
		path = f'plots/sim{sim}'
		if os.path.exists(path):
			shutil.rmtree(path)
		os.makedirs(path)

		Eval.computeOnlyOnce(sim)
		# Timesteps used for evaluation
		for time in range(first_t, last_t):
			Eval.timeStep(sim, time, plot_intermediate_fields, save_plots, show_plots, apply_filter, apply_deltaU_change_wgt, flatten_data)

		n_ts = last_sim - first_sim
		# Errors for each simulation
		BIAS_value = np.mean(Eval.pred_minus_true[-n_ts:]) * 100
		RMSE_value = np.sqrt(np.mean(Eval.pred_minus_true_squared[-n_ts:])) * 100
		STDE_value = np.sqrt( RMSE_value**2 - BIAS_value**2 )

		BIAS_block = np.mean(Eval.pred_minus_true_block[-n_ts:]) * 100
		RSME_block = np.sqrt(np.mean(Eval.pred_minus_true_squared_block[-n_ts:])) * 100
		STDE_block = np.sqrt( RSME_block**2 - BIAS_block**2 )

		BIAS_value_deltap_crude = np.mean(Eval.pred_minus_true_deltap_crude[-n_ts:]) * 100
		RMSE_value_deltap_crude = np.sqrt(np.mean(Eval.pred_minus_true_squared_deltap_crude[-n_ts:])) * 100
		STDE_value_deltap_crude = np.sqrt( RMSE_value_deltap_crude**2 - BIAS_value_deltap_crude**2 )

		BIAS_value_p = np.mean(Eval.pred_minus_true_p) * 100
		RMSE_value_p = np.sqrt(np.mean(Eval.pred_minus_true_squared_p)) * 100
		STDE_value_p = np.sqrt( RMSE_value_p**2 - BIAS_value_p**2 )

		print(f"Eval.pred_minus_true_block len: {len(Eval.pred_minus_true_block)}")
		print(f"Eval.pred_minus_true_block len: {len(Eval.pred_minus_true_squared_block)}")
		print(f'''
			Average error in SIM {sim}:

			** Error in delta_p **

			BIAS: {BIAS_value:.3f}%
			STDE: {STDE_value:.3f}%
			RMSE: {RMSE_value:.3f}%

			** Error in p **

			BIAS: {BIAS_value_p:.5f}%
			STDE: {STDE_value_p:.5f}%
			RMSE: {RMSE_value_p:.5f}%
			''')
		
		if apply_deltaU_change_wgt:
			print(f'''
			Before Weighting:

				** Error in delta_p (blocks) **

				BIAS: {BIAS_block:.3f}%
				STDE: {STDE_block:.3f}%
				RMSE: {RSME_block:.3f}%

				** Error in delta_p **

				BIAS: {BIAS_value_deltap_crude:.3f}%
				STDE: {STDE_value_deltap_crude:.3f}%
				RMSE: {RMSE_value_deltap_crude:.3f}%
			''')


	# Errors for the whole set of simulations
	BIAS_value = np.mean(Eval.pred_minus_true) * 100
	RMSE_value = np.sqrt(np.mean(Eval.pred_minus_true_squared)) * 100
	STDE_value = np.sqrt( RMSE_value**2 - BIAS_value**2 )

	BIAS_value_p = np.mean(Eval.pred_minus_true_p) * 100
	RMSE_value_p = np.sqrt(np.mean(Eval.pred_minus_true_squared_p)) * 100
	STDE_value_p = np.sqrt( RMSE_value_p**2 - BIAS_value_p**2 )

	BIAS_block = np.mean(Eval.pred_minus_true_block) * 100
	RSME_block = np.sqrt(np.mean(Eval.pred_minus_true_squared_block)) * 100
	STDE_block = np.sqrt( RSME_block**2 - BIAS_block**2 )

	BIAS_value_deltap_crude = np.mean(Eval.pred_minus_true_deltap_crude) * 100
	RMSE_value_deltap_crude = np.sqrt(np.mean(Eval.pred_minus_true_squared_deltap_crude)) * 100
	STDE_value_deltap_crude = np.sqrt( RMSE_value**2 - BIAS_value**2 )

	print(f'''
	Average across the WHOLE set of simulations:

	** Error in delta_p **

	BIAS: {BIAS_value:.3f}%
	STDE: {STDE_value:.3f}%
	RMSE: {RMSE_value:.3f}%

	** Error in p **

	BIAS: {BIAS_value_p:.5f}%
	STDE: {STDE_value_p:.5f}%
	RMSE: {RMSE_value_p:.5f}%
	
	Before Weighting:

		** Error in delta_p (blocks) **

		BIAS: {BIAS_block:.3f}%
		STDE: {STDE_block:.3f}%
		RMSE: {RSME_block:.3f}%

		** Error in delta_p **

		BIAS: {BIAS_value_deltap_crude:.3f}%
		STDE: {STDE_value_deltap_crude:.3f}%
		RMSE: {RMSE_value_deltap_crude:.3f}%

	''', flush = True)

	if create_GIF:
		n_sims = last_sim - first_sim + 1
		n_ts = last_t - first_t
		visualization.createGIF(n_sims, n_ts)


if __name__ == '__main__':

	delta = 5e-3
	model_name = 'model_small-std-0.95.h5'
	shape = 128
	overlap_ratio = 0.25
	dataset_path = '../dataset_plate_deltas_5sim20t.hdf5' #adjust dataset path
	standardization_method = 'std'

	plot_intermediate_fields = True
	save_plots = True
	show_plots = False
	apply_filter = False
	create_GIF = True

	first_sim = 0
	last_sim = 2
	first_t = 0
	last_t = 5

	call_SM_main(delta, model_name, shape, overlap_ratio, dataset_path,	\
				plot_intermediate_fields, standardization_method, save_plots, show_plots, apply_filter, create_GIF, \
				first_sim, last_sim, first_t, last_t)

