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
mpi4py.rc.initialize = False  # Don't call MPI_Init (OpenFOAM already did)
mpi4py.rc.finalize = False
from mpi4py import MPI

# Manually attach to the already-initialized MPI environment
if not MPI.Is_initialized():
    MPI.Init()

import pickle as pk
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import tensorly as tl

from pressure_SM_delta_delta._3D.CFD_usable.utils import memory
from pressure_SM_delta_delta._3D.train_and_eval.utils.data_processing import interpolate_fill_njit, interp_weights, create_uniform_grid, _unpack_grid_res
from pressure_SM_delta_delta._3D.train_and_eval.utils.domain_geometry import domain_dist
from pressure_SM_delta_delta._3D.train_and_eval.utils.model_utils import define_model_arch
from pressure_SM_delta_delta._3D.train_and_eval.utils import data_processing as utils_data

from pressure_SM_delta_delta._3D.train_and_eval.assembly import assemble_prediction
from pressure_SM_delta_delta._3D.train_and_eval.neural_networks import (
    MLP, SimpleCNN3D, Simple_multi_layer_3D, FNO3d, GNN, MLP_Mixer_3D, 
    UNet3D, UNet3D_deep, UNet3D_attention, SymmetricPadding3D, SimpleCNN3D_two_heads
)


_inspect_call_count = 0


def _plot_two_head_decomposition_multiZ(
	p_smooth_grid,
	p_local_grid,
	call_count,
	output_dir,
	z_slices=[0, 20, 40, 60, 80, 95],
):
	"""
	Plot multi-z-slice analysis of the assembled p_smooth and p_local grids
	from the two-head CNN model. Saves one figure per head.

	z_slices: list of percentages (0-100) of the z-axis to slice.
	"""
	import matplotlib.pyplot as plt

	def _safe_vmax(arr, fallback=1e-8):
		v = float(np.nanmax(np.abs(arr)))
		return v if np.isfinite(v) and v > 0.0 else fallback

	z_len = p_smooth_grid.shape[0]
	indices = [int((s / 100) * z_len) for s in z_slices]
	n_panels = len(indices)

	os.makedirs(output_dir, exist_ok=True)

	for field, label, fname_key in [
		(p_smooth_grid, "p_smooth (far-field head)", "p_smooth"),
		(p_local_grid,  "p_local  (obstacle head)",  "p_local"),
	]:
		fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.5 * n_panels), squeeze=False)
		axes = axes.flatten()

		for panel_idx, z_idx in enumerate(indices):
			slice_data = field[z_idx, :, :].copy()
			vmax = _safe_vmax(slice_data)
			im = axes[panel_idx].imshow(
				slice_data, origin="lower", aspect="auto",
				cmap="RdBu_r", vmin=-vmax, vmax=vmax,
			)
			axes[panel_idx].set_title(f"{label} - z-slice {z_idx}", fontsize=18, fontweight='bold')
			axes[panel_idx].set_xlabel("x")
			axes[panel_idx].set_ylabel("y")
			plt.colorbar(im, ax=axes[panel_idx])
			axes[panel_idx].axis("off")

		plt.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.08, hspace=0.3)
		title = "p_smooth" if "smooth" in fname_key else "p_local"
		fig.suptitle(
			f"{title} Multi-Z-Slice Analysis - Call {call_count}",
			fontsize=22, fontweight="bold",
		)
		fname = os.path.join(output_dir, f"{fname_key}_slices_{call_count:05d}.png")
		plt.savefig(fname, dpi=100, bbox_inches="tight")
		plt.close(fig)
		print(f"[two_heads] Saved: {fname}")


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
	spatial_tucker_ranks,
	verbose=True,
	inspect_results=False,
	inspect_output_dir='SM_inspect',
	use_feature_decomposition=True,
	add_ddu_input=True,
	add_dddu_input=True,
	add_U_input=False,
	add_dU_input=False,
	add_dp_prev_input=False,
	add_p_prev_input=False,
	add_ddp_prev_input=False,
	add_div_ddu_input=False,
	add_div_du_input=False,
	add_div_u_input=False,
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

	global use_feature_decomposition_g
	use_feature_decomposition_g = use_feature_decomposition

	# Store these so init_func can rebuild the CNN with the correct block shape
	global dropout_rate_g, regularization_g, n_layers_g, width_g, weights_fn_g
	dropout_rate_g = dropout_rate
	regularization_g = regularization
	weights_fn_g = weights_fn
	n_layers_g, width_g = define_model_arch(model_arch)
	global effective_model_arch_g
	_FLAT_MODELS_EARLY = ('mlp_small', 'mlp_big', 'mlp_small_unet', 'mlp_huge', 'mlp_huger', 'conv1d', 'mlp_attention')
	effective_model_arch_g = 'cnn' if (not use_feature_decomposition and model_arch.lower() in _FLAT_MODELS_EARLY) else model_arch

	global inspect_results_g, inspect_output_dir_g
	inspect_results_g = inspect_results
	inspect_output_dir_g = inspect_output_dir

	global comm, rank, nprocs
	print('Initializing MPI communication in Python')
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	nprocs = comm.Get_size()

	if rank == 0:
		if inspect_results:
			os.makedirs(inspect_output_dir, exist_ok=True)
			print(f'[inspect_results] Output directory: {os.path.abspath(inspect_output_dir)}')
		global in_factors, out_factors
		if use_feature_decomposition:
			print('Loading the Tucker factors')
			with open(tucker_fn, 'rb') as f:
				tucker_data = pk.load(f)
			in_factors = tucker_data['input_factors']
			out_factors = tucker_data['output_factors']

			# ---- NEW: pretranspose factors for fast einsum paths (float32 + contiguous) ----
			# x_array has shape (N, bs, bs, bs, 7) — [ddU_x, ddU_y, ddU_z, dddU_x, dddU_y, dddU_z, sdf]
			# We need transpose=True for modes [1,2,3,4] => multiply by factor.T along each mode.
			global in_factors_T, out_factors_c
			in_factors_T = [None]  # keep 1-based indexing consistent with your existing list
			for k in range(1, 5):
				F = np.asarray(in_factors[k], dtype=np.float64)
				in_factors_T.append(np.ascontiguousarray(F.T))

			# For inverse Tucker (transpose=False), we want non-transposed factors contiguous.
			out_factors_c = [None]
			for k in range(1, 4):
				F = np.asarray(out_factors[k], dtype=np.float64)
				out_factors_c.append(np.ascontiguousarray(F))
			# ------------------------------------------------------------------------------
		else:
			print('[load_tucker_and_NN] use_feature_decomposition=False: skipping Tucker factor loading.')

		## Loading values for blocks normalization
		if maxs_fn.endswith('.npy'):	
			maxs = np.load(maxs_fn)
		else:
			maxs = np.loadtxt(maxs_fn)

		global max_abs_U_x, max_abs_U_y, max_abs_U_z, max_abs_dU_x, max_abs_dU_y, max_abs_dU_z, max_abs_ddU_x, max_abs_ddU_y, max_abs_ddU_z, max_abs_dddU_x, max_abs_dddU_y, max_abs_dddU_z, max_abs_dist, max_abs_delta_delta_p, max_abs_delta_p_prev, max_abs_ddp_prev, max_abs_p_prev, max_abs_div_ddu, max_abs_div_du, max_abs_div_u
		global add_U_input_g, add_dU_input_g, add_ddu_input_g, add_dddu_input_g, add_dp_prev_input_g, add_p_prev_input_g, add_ddp_prev_input_g, add_div_ddu_input_g, add_div_du_input_g, add_div_u_input_g
		add_U_input_g = add_U_input
		add_dU_input_g = add_dU_input
		add_ddu_input_g = add_ddu_input
		add_dddu_input_g = add_dddu_input
		add_dp_prev_input_g = add_dp_prev_input
		add_p_prev_input_g = add_p_prev_input
		add_ddp_prev_input_g = add_ddp_prev_input
		add_div_ddu_input_g = add_div_ddu_input
		add_div_du_input_g = add_div_du_input
		add_div_u_input_g = add_div_u_input
		
		# Parse maxs based on flags: [U if add_U] [dU if add_dU] [ddU if add_ddu] dddU dist [p_prev if add_p_prev] [dp_prev if use_prev_dp] [ddp_prev if add_ddp_prev] ddp
		ch_idx = 0
		if add_U_input:
			max_abs_U_x, max_abs_U_y, max_abs_U_z = maxs[ch_idx:ch_idx+3]
			ch_idx += 3
		else:
			max_abs_U_x = max_abs_U_y = max_abs_U_z = 1.0  # unused

		if add_dU_input:
			max_abs_dU_x, max_abs_dU_y, max_abs_dU_z = maxs[ch_idx:ch_idx+3]
			ch_idx += 3
		else:
			max_abs_dU_x = max_abs_dU_y = max_abs_dU_z = 1.0  # unused
		
		if add_ddu_input:
			max_abs_ddU_x, max_abs_ddU_y, max_abs_ddU_z = maxs[ch_idx:ch_idx+3]
			ch_idx += 3
		else:
			max_abs_ddU_x = max_abs_ddU_y = max_abs_ddU_z = 1.0  # unused
		
		if add_dddu_input:
			max_abs_dddU_x, max_abs_dddU_y, max_abs_dddU_z = maxs[ch_idx:ch_idx+3]
			ch_idx += 3
		else:
			max_abs_dddU_x = max_abs_dddU_y = max_abs_dddU_z = 1.0  # unused
		max_abs_dist = maxs[ch_idx]
		ch_idx += 1
		if add_p_prev_input:
			max_abs_p_prev = maxs[ch_idx]
			ch_idx += 1
		else:
			max_abs_p_prev = 1.0  # unused
		if add_dp_prev_input:
			max_abs_delta_p_prev = maxs[ch_idx]
			ch_idx += 1
		else:
			max_abs_delta_p_prev = 1.0  # unused
		if add_ddp_prev_input:
			max_abs_ddp_prev = maxs[ch_idx]
			ch_idx += 1
		else:
			max_abs_ddp_prev = 1.0  # unused
		if add_div_ddu_input:
			max_abs_div_ddu = maxs[ch_idx]
			ch_idx += 1
		else:
			max_abs_div_ddu = 1.0  # unused
		if add_div_du_input:
			max_abs_div_du = maxs[ch_idx]
			ch_idx += 1
		else:
			max_abs_div_du = 1.0  # unused
		if add_div_u_input:
			max_abs_div_u = maxs[ch_idx]
			ch_idx += 1
		else:
			max_abs_div_u = 1.0  # unused
		max_abs_delta_delta_p = maxs[ch_idx]
		
		# Loading values for standardization
		data = np.load(std_vals_fn)
		global mean_in, std_in, mean_out, std_out
		mean_in = data['mean_in']
		std_in = data['std_in']
		mean_out = data['mean_out']
		std_out = data['std_out']

		# Auto-calculate last_tucker_rank if not provided
		# Base is 4; each input flag adds 3, add_dp_prev_input adds 1: [U if add_U_input] [ddU if add_ddu_input] dddU sdf delta_p [delta_p_prev if add_dp_prev_input] -> mapped through Tucker
		# Actually for CNN (use_feature_decomposition=False), it's: [U if add_U_input] [ddU if add_ddu_input] dddU [+delta_p_prev if add_dp_prev_input] -> input channels
		# For Tucker decomposition (use_feature_decomposition=True), it uses the last Tucker factor dimension
		last_tucker_rank = 1  # base: sdf only
		if add_U_input:
			last_tucker_rank += 3
		if add_dU_input:
			last_tucker_rank += 3
		if add_ddu_input:
			last_tucker_rank += 3
		if add_dddu_input:
			last_tucker_rank += 3
		if add_p_prev_input:
			last_tucker_rank += 1
		if add_dp_prev_input:
			last_tucker_rank += 1
		if add_ddp_prev_input:
			last_tucker_rank += 1
		if add_div_ddu_input:
			last_tucker_rank += 1
		if add_div_du_input:
			last_tucker_rank += 1
		if add_div_u_input:
			last_tucker_rank += 1
		if verbose:
			print(f'[load_tucker_and_NN] Auto-calculated last_tucker_rank: {last_tucker_rank}')
	
		# Store Tucker rank info so init_func can build the MLP with correct sizes
		global spatial_tucker_ranks_g, last_tucker_rank_g
		spatial_tucker_ranks_g = spatial_tucker_ranks
		last_tucker_rank_g = last_tucker_rank

		if verbose:
			print(f'[load_tucker_and_NN] Configuration stored. Model will be created in init_func.')
			print(f'[load_tucker_and_NN] overlap_ratio: {overlap_ratio}')
			print(f'[load_tucker_and_NN] add_U_input: {add_U_input}, add_dU_input: {add_dU_input}, add_ddu_input: {add_ddu_input}, add_dddu_input: {add_dddu_input}, add_dp_prev_input: {add_dp_prev_input}, add_p_prev_input: {add_p_prev_input}, add_ddp_prev_input: {add_ddp_prev_input}, add_div_ddu_input: {add_div_ddu_input}, add_div_du_input: {add_div_du_input}, add_div_u_input: {add_div_u_input}, last_tucker_rank: {last_tucker_rank}')


def reload_weights(weights_fn):
	"""Reload NN weights from disk after incremental retraining.

	This is called by the solver after each retrain cycle so that the
	in-memory model is always up to date without restarting the simulation.
	Only the model weights are refreshed; Tucker factors and interpolation
	data remain unchanged.
	"""
	import os, time
	global model, mean_in, std_in, mean_out, std_out
	if rank == 0:
		if os.path.exists(weights_fn):
			fsize = os.path.getsize(weights_fn)
			mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(weights_fn)))
			print(f"[reload_weights] Loading: {weights_fn}  |  size={fsize} bytes  |  modified={mtime}")
		else:
			print(f"[reload_weights] ERROR: weights file not found: {weights_fn}")
			return
		# Checksum of first layer weights before reload
		import numpy as np
		weights_before = model.get_weights()
		chk_before = sum(float(np.sum(w)) for w in weights_before)
		model.load_weights(weights_fn)
		weights_after = model.get_weights()
		chk_after = sum(float(np.sum(w)) for w in weights_after)
		print(f"[reload_weights] Weight checksum before={chk_before:.6f}  after={chk_after:.6f}  changed={not np.isclose(chk_before, chk_after)}")
		print("[reload_weights] Model weights reloaded successfully.")
		# Reload normalization stats so inference always uses the same stats
		# as those used when the model was trained.
		std_vals_fn = os.path.join(os.path.dirname(weights_fn), 'mean_std.npz')
		if os.path.exists(std_vals_fn):
			norm_data = np.load(std_vals_fn)
			mean_in  = norm_data['mean_in'];  std_in  = norm_data['std_in']
			mean_out = norm_data['mean_out']; std_out = norm_data['std_out']
			print(f"[reload_weights] Normalization stats reloaded from {std_vals_fn}")
		else:
			print(f"[reload_weights] WARNING: normalization file not found: {std_vals_fn}")


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
	dx, dy, dz = _unpack_grid_res(grid_res)

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

		# Use min/max across all relevant arrays for robust domain bounding
		all_x = [array_concat[...,3]]
		all_y = [array_concat[...,4]]
		all_z = [array_concat[...,5]]
		# Add obstacle and boundary arrays if they have the right shape
		for arr in [obst, y_bot, y_top, z_bot, z_top]:
			arr = np.asarray(arr)
			if arr.ndim > 1 and arr.shape[-1] >= 3:
				all_x.append(arr[...,0])
				all_y.append(arr[...,1])
				all_z.append(arr[...,2])

		limits = {
			'x_min': np.min(np.concatenate(all_x)),
			'x_max': np.max(np.concatenate(all_x)),
			'y_min': np.min(np.concatenate(all_y)),
			'y_max': np.max(np.concatenate(all_y)),
			'z_min': np.min(np.concatenate(all_z)),
			'z_max': np.max(np.concatenate(all_z))
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
		grid_shape_x, grid_shape_y, grid_shape_z = utils_data.get_grid_shape(limits, grid_res)

		x0 = np.min(X0)
		y0 = np.min(Y0)
		z0 = np.min(Z0)

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

		# CHANGED: precompute and cache indices as int32 arrays for faster indexing
		global indices_i, indices_j, indices_k
		indices_i = indices[:, 0].astype(np.int32)
		indices_j = indices[:, 1].astype(np.int32)
		indices_k = indices[:, 2].astype(np.int32)

		# --- Determine actual block sizes now that grid shape is known ---
		# This replicates the same clamping logic used in py_func so the CNN is
		# built with the correct (potentially non-cubic) spatial shape.
		block_size = block_size_g
		overlap = int(overlap_ratio_g * block_size)
		_n_z = 1 if grid_shape_z <= block_size else int(np.ceil((grid_shape_z - block_size) / (block_size - overlap))) + 1
		_n_y = 1 if grid_shape_y <= block_size else int(np.ceil((grid_shape_y - block_size) / (block_size - overlap))) + 1
		_n_x = 1 if grid_shape_x <= block_size else int(np.ceil((grid_shape_x - block_size) / (block_size - overlap))) + 1
		global block_size_z_g, block_size_y_g, block_size_x_g
		block_size_z_g = grid_shape_z if _n_z == 1 else block_size
		block_size_y_g = grid_shape_y if _n_y == 1 else block_size
		block_size_x_g = grid_shape_x if _n_x == 1 else block_size
		print(f'[init_func] effective block size: z={block_size_z_g}, y={block_size_y_g}, x={block_size_x_g}')

		# Build model now that the true block shape is known.
		# This is done here for BOTH paths (Tucker+MLP and raw+CNN) so that
		# load_tucker_and_NN stays free of any model instantiation.
		global model
		if use_feature_decomposition_g:
			n_layers, width = n_layers_g, width_g
			input_features_size = spatial_tucker_ranks_g[0] * spatial_tucker_ranks_g[1] * spatial_tucker_ranks_g[2] * last_tucker_rank_g
			output_features_size = spatial_tucker_ranks_g[0] * spatial_tucker_ranks_g[1] * spatial_tucker_ranks_g[2]
			print(f'[init_func] Creating MLP model (in={input_features_size}, out={output_features_size})')
			model = MLP(n_layers, width, input_features_size, output_features_size, dropout_rate_g, regularization_g)
		else:
			_bs = (block_size_z_g, block_size_y_g, block_size_x_g)
			arch = effective_model_arch_g.lower()
			print(f'[init_func] Creating {arch} model with block shape {_bs}')
			if arch == 'cnn':
				model = SimpleCNN3D(_bs, in_channels=last_tucker_rank_g, dropout_rate=dropout_rate_g, regularization=regularization_g)
			elif arch == 'cnn_two_heads':
				model = SimpleCNN3D_two_heads(_bs, in_channels=last_tucker_rank_g,
								  return_heads=True,
								  dropout_rate=dropout_rate_g, regularization=regularization_g)
			elif arch == 'multi_layer_3d':
				model = Simple_multi_layer_3D(_bs, in_channels=last_tucker_rank_g, n_layers=n_layers_g, width=width_g,
				                              dropout_rate=dropout_rate_g, regularization=regularization_g)
			elif arch == 'fno3d':
				model = FNO3d(_bs, in_channels=last_tucker_rank_g)
			elif arch == 'gnn':
				model = GNN(_bs)  # GNN does not expose in_channels
			elif arch == 'mixer':
				model = MLP_Mixer_3D(n_layers_g, _bs, in_channels=last_tucker_rank_g, dropout_rate=dropout_rate_g, regularization=regularization_g)
			elif arch == 'unet3d':
				model = UNet3D(_bs, in_channels=last_tucker_rank_g, dropout_rate=dropout_rate_g, regularization=regularization_g)
			elif arch == 'unet3d_deep':
				model = UNet3D_deep(_bs, in_channels=last_tucker_rank_g, dropout_rate=dropout_rate_g, regularization=regularization_g)
			elif arch == 'unet3d_attention':
				model = UNet3D_attention(_bs, in_channels=last_tucker_rank_g, dropout_rate=dropout_rate_g, regularization=regularization_g)
			else:
				print(f'[init_func] Unknown arch "{arch}", falling back to cnn.')
				model = SimpleCNN3D(_bs, in_channels=last_tucker_rank_g, dropout_rate=dropout_rate_g, regularization=regularization_g)
		model.load_weights(weights_fn_g)
		print(f'[init_func] Model loaded from {weights_fn_g}')

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
	
	Channel Layout in grid and x_array:
	When all inputs are enabled (expected 16 channels):
	  Channels 0-2:   U (velocity, if add_U_input_g=True)
	  Channels 3-5:   dU (velocity increment, if add_dU_input_g=True)
	  Channels 6-8:   ddU (second velocity increment, if add_ddu_input_g=True)
	  Channels 9-11:  dddU (velocity difference, if add_dddu_input_g=True)
	  Channel 12:     p_prev (previous pressure, if add_p_prev_input_g=True)
	  Channel 13:     dp_prev (previous pressure increment, if add_dp_prev_input_g=True)
	  Channel 14:     ddp_prev (second pressure increment, if add_ddp_prev_input_g=True)
	  Channel 15:     sdf (signed distance function, always included as last channel)
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
		# Extract data based on flags
		# Array layout (15 cols): 0-2 U, 3-5 ddU, 6-8 ddU_prev, 9 ddp_prev, 10 dp_prev, 11-13 dU, 14 p_rgh_prev
		ch_idx = 0
		if add_U_input_g:
			U = array[..., 0:3]
			ch_idx += 3
		else:
			U = None
		
		delta_delta_U = array[..., 3:6]
		delta_delta_U_prev = array[..., 6:9]
		if add_ddu_input_g:
			ch_idx += 3
		
		# dddU (optional)
		delta_delta_U_diff = delta_delta_U - delta_delta_U_prev
		if add_dddu_input_g:
			ch_idx += 3
		
		# Channel 9: delta_delta_p_prev (SM input if add_ddp_prev_input, else not used)
		delta_delta_p_prev = array[..., 9:10]

		# Channel 10: delta_p_prev — SM input (if add_dp_prev_input)
		if add_dp_prev_input_g:
			delta_p_prev = array[..., 10:11]
			ch_idx += 1
		else:
			delta_p_prev = None

		# Channels 11-13: delta_U (first velocity increment)
		if add_dU_input_g:
			delta_U = array[..., 11:14]
			ch_idx += 3
		else:
			delta_U = None

		# Channel 14: p_rgh_prev (absolute previous pressure)
		if add_p_prev_input_g:
			p_rgh_prev = array[..., 14:15]
			ch_idx += 1
		else:
			p_rgh_prev = None

		if add_div_ddu_input_g:
			div_ddu = array[..., 15:16]
			ch_idx += 1
		else:
			div_ddu = None
		
		if add_div_du_input_g:
			div_du = array[..., 16:17]
			ch_idx += 1
		else:
			div_du = None

		if add_div_u_input_g:
			div_u = array[..., 17:18]
			ch_idx += 1
		else:
			div_u = None
		

		delta_U_changed = np.abs(delta_delta_U).sum(axis=-1)/np.abs(delta_delta_U).max()
		delta_delta_U_changed = np.abs(delta_delta_U - delta_delta_U_prev).sum(axis=-1)
        
		#delta_delta_U_changed = np.abs(delta_delta_U - delta_delta_U_prev).sum(axis=-1)
		#if delta_delta_U_changed.max() > 0:
		#	delta_delta_U_changed /= delta_delta_U_changed.max()

		# Normalize by U_max_norm
		if add_U_input_g:
			U_adim = U / U_max_norm  # U normalized
		if add_dU_input_g:
			delta_U_adim = delta_U / U_max_norm  # dU normalized
		if add_ddu_input_g:
			delta_delta_U_adim = delta_delta_U / U_max_norm  # ddU normalized
		if add_dddu_input_g:
			delta_delta_U_diff_adim = delta_delta_U_diff / U_max_norm  # dddU normalized
		else:
			delta_delta_U_diff_adim = None
		
		# Normalize delta_p_prev (SM input) if present
		if add_dp_prev_input_g:
			delta_p_prev_adim = delta_p_prev / (U_max_norm ** 2.0)
		else:
			delta_p_prev_adim = None

		# Normalize p_rgh_prev if present
		if add_p_prev_input_g:
			p_rgh_prev_adim = p_rgh_prev / (U_max_norm ** 2.0)
		else:
			p_rgh_prev_adim = None

		# Normalize delta_delta_p_prev (SM input) if add_ddp_prev_input is enabled
		if add_ddp_prev_input_g:
			delta_ddp_prev_adim = delta_delta_p_prev / (U_max_norm ** 2.0)
		else:
			delta_ddp_prev_adim = None
		
		if add_div_ddu_input_g:
			div_ddu_adim = div_ddu / U_max_norm
		else:
			div_ddu_adim = None

		if add_div_du_input_g:
			div_du_adim = div_du / U_max_norm
		else:
			div_du_adim = None
		
		if add_div_u_input_g:
			div_u_adim = div_u / U_max_norm
		else:
			div_u_adim = None

		if verbose_g: 
			print(f"Data pre-processing: {time.time()-t0} s")

		t0 = time.time()

		# Interpolate all components based on flags
		if add_U_input_g:
			U_x_interp = interpolate_fill_njit(U_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
			U_y_interp = interpolate_fill_njit(U_adim[:, 1], vert_OFtoNP_array, weights_OFtoNP_array)
			U_z_interp = interpolate_fill_njit(U_adim[:, 2], vert_OFtoNP_array, weights_OFtoNP_array)

		if add_dU_input_g:
			dU_x_interp = interpolate_fill_njit(delta_U_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
			dU_y_interp = interpolate_fill_njit(delta_U_adim[:, 1], vert_OFtoNP_array, weights_OFtoNP_array)
			dU_z_interp = interpolate_fill_njit(delta_U_adim[:, 2], vert_OFtoNP_array, weights_OFtoNP_array)
		
		if add_ddu_input_g:
			ddU_x_interp = interpolate_fill_njit(delta_delta_U_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
			ddU_y_interp = interpolate_fill_njit(delta_delta_U_adim[:, 1], vert_OFtoNP_array, weights_OFtoNP_array)
			ddU_z_interp = interpolate_fill_njit(delta_delta_U_adim[:, 2], vert_OFtoNP_array, weights_OFtoNP_array)
		
		if add_dddu_input_g:
			dddU_x_interp = interpolate_fill_njit(delta_delta_U_diff_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
			dddU_y_interp = interpolate_fill_njit(delta_delta_U_diff_adim[:, 1], vert_OFtoNP_array, weights_OFtoNP_array)
			dddU_z_interp = interpolate_fill_njit(delta_delta_U_diff_adim[:, 2], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			dddU_x_interp = dddU_y_interp = dddU_z_interp = None
		delta_U_changed_interp = interpolate_fill_njit(delta_U_changed, vert_OFtoNP_array, weights_OFtoNP_array)
		
		# Interpolate delta_p_prev (SM input) and delta_delta_p_prev
		if add_dp_prev_input_g:
			delta_p_prev_interp = interpolate_fill_njit(delta_p_prev_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			delta_p_prev_interp = None
		delta_delta_p_prev_interp = interpolate_fill_njit(delta_delta_p_prev[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
		if add_ddp_prev_input_g:
			delta_ddp_prev_interp = interpolate_fill_njit(delta_ddp_prev_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			delta_ddp_prev_interp = None

		if add_p_prev_input_g:
			p_rgh_prev_interp = interpolate_fill_njit(p_rgh_prev_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			p_rgh_prev_interp = None

		if add_div_ddu_input_g:
			div_ddu_interp = interpolate_fill_njit(div_ddu_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			div_ddu_interp = None
		
		if add_div_du_input_g:
			div_du_interp = interpolate_fill_njit(div_du_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			div_du_interp = None

		if add_div_u_input_g:
			div_u_interp = interpolate_fill_njit(div_u_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			div_u_interp = None

		if verbose_g:
			print(f"1st interpolation took: {time.time()-t0} s")

		t0 = time.time()

		# Grid channels: [U(3 if add_U)] [dU(3 if add_dU)] [ddU(3 if add_ddu)] [dddU(3 if add_dddu)] [p_prev(1 if add_p_prev)] [dp_prev(1 if add_dp_prev)] [ddp_prev(1 if add_ddp_prev)] sdf(1)
		n_grid_ch = (3 if add_U_input_g else 0) + (3 if add_dU_input_g else 0) + (3 if add_ddu_input_g else 0) + (3 if add_dddu_input_g else 0) + (1 if add_p_prev_input_g else 0) + (1 if add_dp_prev_input_g else 0) + (1 if add_ddp_prev_input_g else 0) + (1 if add_div_ddu_input_g else 0) + (1 if add_div_du_input_g else 0) + (1 if add_div_u_input_g else 0) + 1
		grid = np.zeros((grid_shape_z, grid_shape_y, grid_shape_x, n_grid_ch), dtype=np.float64)
		delta_U_change_grid = np.zeros((grid_shape_z, grid_shape_y, grid_shape_x), dtype=np.float64)

		# Build the interpolated stack based on what's included
		ch_idx = 0
		interp_parts = []
		if add_U_input_g:
			interp_parts.append(np.column_stack([U_x_interp, U_y_interp, U_z_interp]))
			ch_idx += 3
		if add_dU_input_g:
			interp_parts.append(np.column_stack([dU_x_interp, dU_y_interp, dU_z_interp]))
			ch_idx += 3
		if add_ddu_input_g:
			interp_parts.append(np.column_stack([ddU_x_interp, ddU_y_interp, ddU_z_interp]))
			ch_idx += 3
		if add_dddu_input_g:
			interp_parts.append(np.column_stack([dddU_x_interp, dddU_y_interp, dddU_z_interp]))
			ch_idx += 3
		if add_p_prev_input_g:
			interp_parts.append(p_rgh_prev_interp[:, np.newaxis])  # p_prev channel
			ch_idx += 1
		if add_dp_prev_input_g:
			interp_parts.append(delta_p_prev_interp[:, np.newaxis])  # delta_p_prev channel
			ch_idx += 1
		if add_ddp_prev_input_g:
			interp_parts.append(delta_ddp_prev_interp[:, np.newaxis])  # delta_delta_p_prev channel
			ch_idx += 1
		if add_div_ddu_input_g:
			interp_parts.append(div_ddu_interp[:, np.newaxis])  # div_ddu channel
			ch_idx += 1
		if add_div_du_input_g:
			interp_parts.append(div_du_interp[:, np.newaxis])  # div_du channel
			ch_idx += 1
		if add_div_u_input_g:
			interp_parts.append(div_u_interp[:, np.newaxis])  # div_u channel
			ch_idx += 1
		interp_stack = np.column_stack(interp_parts)
		grid[indices_i, indices_j, indices_k, :ch_idx] = interp_stack
		grid[:, :, :, ch_idx] = sdfunct[:, :, :, 0]  # sdf is at position ch_idx


		fig, axs = plt.subplots(2, 3, figsize=(18, 10))
		mid_z = grid_shape_z // 2
		mid_y = grid_shape_y // 2
		mid_x = grid_shape_x // 2

		# delta_delta_Ux, Uy, Uz slices (always at first 3 channels if U is not present, or after U if present)
		u_offset = 3 if add_U_input_g else 0
		ddu_offset = u_offset + (3 if add_dU_input_g else 0) + (3 if add_ddu_input_g else 0)
		axs[0, 0].imshow(grid[mid_z, :, :, ddu_offset], aspect='auto')
		axs[0, 0].set_title('delta_delta_Ux (z mid-slice)')
		axs[0, 1].imshow(grid[:, mid_y, :, ddu_offset+1], aspect='auto')
		axs[0, 1].set_title('delta_delta_Uy (y mid-slice)')
		axs[0, 2].imshow(grid[:, :, mid_x, ddu_offset+2], aspect='auto')
		axs[0, 2].set_title('delta_delta_Uz (x mid-slice)')

		# (delta_delta_U - delta_delta_U_prev) norm (dddU magnitude)
		diff_norm = np.linalg.norm(grid[..., ddu_offset:ddu_offset+3], axis=-1)
		axs[1, 0].imshow(diff_norm[mid_z, :, :], aspect='auto')
		axs[1, 0].set_title('||dddU|| (z mid-slice)')

		# sdf (always at position n_grid_ch-1, the last channel)
		sdf_ch_idx = n_grid_ch - 1
		axs[1, 1].imshow(grid[mid_z, :, :, sdf_ch_idx], aspect='auto')
		axs[1, 1].set_title('sdf (z mid-slice)')

		# delta_p_prev if present (comes before sdf)
		if add_dp_prev_input_g:
			dp_prev_ch_idx = (3 if add_U_input_g else 0) + (3 if add_dU_input_g else 0) + (3 if add_ddu_input_g else 0) + (3 if add_dddu_input_g else 0) + (1 if add_p_prev_input_g else 0)
			axs[1, 2].imshow(grid[mid_z, :, :, dp_prev_ch_idx], aspect='auto')
			axs[1, 2].set_title('delta_p_prev (z mid-slice)')
		else:
			axs[1, 2].imshow(sdfunct[mid_z, :, :, 0], aspect='auto')
			axs[1, 2].set_title('sdfunct (z mid-slice)')

		plt.tight_layout()
		plt.savefig("grid_visualization_step.png")

		delta_U_change_grid[indices_i, indices_j, indices_k] = delta_U_changed_interp

		# Build illustration grid for delta_delta_p_prev (channel 9)
		delta_delta_p_prev_grid = np.zeros((grid_shape_z, grid_shape_y, grid_shape_x), dtype=np.float64)
		delta_delta_p_prev_grid[indices_i, indices_j, indices_k] = delta_delta_p_prev_interp

		# Normalization factors order: [U(3 if add_U)] [dU(3 if add_dU)] [ddU(3 if add_ddu)] dddU(3) [p_prev if add_p_prev] [dp_prev if use_prev_dp] [ddp_prev if add_ddp_prev] [div_ddu if add_div_ddu] [div_du if add_div_du] [div_u if add_div_u] sdf(1)
		norm_parts = []
		if add_U_input_g:
			norm_parts.extend([max_abs_U_x, max_abs_U_y, max_abs_U_z])
		if add_dU_input_g:
			norm_parts.extend([max_abs_dU_x, max_abs_dU_y, max_abs_dU_z])
		if add_ddu_input_g:
			norm_parts.extend([max_abs_ddU_x, max_abs_ddU_y, max_abs_ddU_z])
		if add_dddu_input_g:
			norm_parts.extend([max_abs_dddU_x, max_abs_dddU_y, max_abs_dddU_z])  # dddU
		if add_p_prev_input_g:
			norm_parts.append(max_abs_p_prev)  # p_prev
		if add_dp_prev_input_g:
			norm_parts.append(max_abs_delta_p_prev)  # delta_p_prev
		if add_ddp_prev_input_g:
			norm_parts.append(max_abs_ddp_prev)  # delta_delta_p_prev
		if add_div_ddu_input_g:
			norm_parts.append(max_abs_div_ddu)  # div(delta_delta_U)
		if add_div_du_input_g:
			norm_parts.append(max_abs_div_du)  # div(delta_U)
		if add_div_u_input_g:
			norm_parts.append(max_abs_div_u)  # div(U)
		norm_parts.append(max_abs_dist)  # sdf (always last)
		norm_factors = np.array(norm_parts, dtype=np.float64)
		
		if len(norm_factors) != n_grid_ch:
			raise ValueError(f"Normalization factors count ({len(norm_factors)}) does not match grid channels ({n_grid_ch})")
		grid /= norm_factors[None, None, None, :]

		if verbose_g:
			print(f"Filling grid with shape {grid.shape} took: {time.time()-t0} s")

		t0 = time.time()

		grid[np.isnan(grid)] = 0

		# CHANGED: preallocate x_list and indices_list to avoid Python list appends
		# Block sizes were resolved in init_func; use the cached globals.
		block_size_z = block_size_z_g
		block_size_y = block_size_y_g
		block_size_x = block_size_x_g

		overlap = int(overlap_ratio_g * block_size)
		n_z = 1 if grid_shape_z <= block_size else int(np.ceil((grid_shape_z - block_size) / (block_size - overlap))) + 1
		n_y = 1 if grid_shape_y <= block_size else int(np.ceil((grid_shape_y - block_size) / (block_size - overlap))) + 1
		n_x = 1 if grid_shape_x <= block_size else int(np.ceil((grid_shape_x - block_size) / (block_size - overlap))) + 1

		total_blocks = n_x * n_y * n_z
		x_list = np.empty((total_blocks, block_size_z, block_size_y, block_size_x, n_grid_ch), dtype=np.float64)
		indices_list = np.empty((total_blocks, 3), dtype=np.int32)

		b = 0
		for i in range(n_z):
			if n_z == 1:
				z_0 = 0
			else:
				z_0 = i * block_size - i * overlap if i < n_z - 1 else grid_shape_z - block_size
			z_f = z_0 + block_size_z
			for j in range(n_y):
				if n_y == 1:
					y_0 = 0
				else:
					y_0 = j * block_size - j * overlap if j < n_y - 1 else grid_shape_y - block_size
				y_f = y_0 + block_size_y
				for k in range(n_x):
					x_0 = grid_shape_x - k * block_size_x + k * overlap - block_size_x if k < n_x - 1 else 0
					x_f = x_0 + block_size_x

					# Ensure indices are within bounds
					z_0_clip, z_f_clip = max(z_0, 0), min(z_f, grid_shape_z)
					y_0_clip, y_f_clip = max(y_0, 0), min(y_f, grid_shape_y)
					x_0_clip, x_f_clip = max(x_0, 0), min(x_f, grid_shape_x)

					# If n_z==1 or n_y==1, extract block as-is (no padding in those directions, possibly non-cube)
					if n_z == 1 or n_y == 1:
						block = grid[z_0_clip:z_f_clip, y_0_clip:y_f_clip, x_0_clip:x_f_clip, :n_grid_ch]
						# If block is smaller than expected, pad as needed
						pad_shape = (block_size_z, block_size_y, block_size_x, n_grid_ch)
						sz, sy, sx = block.shape[:3]
						if (sz, sy, sx) != (block_size_z, block_size_y, block_size_x):
							block_padded = np.zeros(pad_shape, dtype=np.float64)
							block_padded[:sz, :sy, :sx, :] = block
							block = block_padded

						# Remove per-block domain mean from pressure input channels (sdf is last: n_grid_ch-1)
						_sdf_ch = n_grid_ch - 1
						_vb = (3 if add_U_input_g else 0) + (3 if add_dU_input_g else 0) + (3 if add_ddu_input_g else 0) + (3 if add_dddu_input_g else 0)
						_dm = block[..., _sdf_ch] != 0
						if _dm.any():
							if add_p_prev_input_g:
								block[..., _vb][_dm] -= np.mean(block[..., _vb][_dm])
							if add_dp_prev_input_g:
								_c = _vb + int(add_p_prev_input_g)
								block[..., _c][_dm] -= np.mean(block[..., _c][_dm])

					else:
						# Prepare block and pad if needed (as before)
						block = np.zeros((block_size_z, block_size_y, block_size_x, n_grid_ch), dtype=np.float64)
						block_z = z_f_clip - z_0_clip
						block_y = y_f_clip - y_0_clip
						block_x = x_f_clip - x_0_clip
						block[:block_z, :block_y, :block_x, :] = grid[z_0_clip:z_f_clip, y_0_clip:y_f_clip, x_0_clip:x_f_clip, :n_grid_ch]

						# Remove per-block domain mean from pressure input channels (sdf is last: n_grid_ch-1)
						_sdf_ch = n_grid_ch - 1
						_vb = (3 if add_U_input_g else 0) + (3 if add_dU_input_g else 0) + (3 if add_ddu_input_g else 0) + (3 if add_dddu_input_g else 0)
						_dm = block[..., _sdf_ch] != 0
						if _dm.any():
							if add_p_prev_input_g:
								block[..., _vb][_dm] -= np.mean(block[..., _vb][_dm])
							if add_dp_prev_input_g:
								_c = _vb + int(add_p_prev_input_g)
								block[..., _c][_dm] -= np.mean(block[..., _c][_dm])
							if add_ddp_prev_input_g:
								_c2 = _c + int(add_dp_prev_input_g)
								block[..., _c2][_dm] -= np.mean(block[..., _c2][_dm])

					x_list[b] = block
					indices_list[b] = [i, j, n_x - 1 - k]
					b += 1

		x_array = x_list  # already an array

		# DEBUG: Verify channel count
		expected_channels = (3 if add_U_input_g else 0) + (3 if add_dU_input_g else 0) + (3 if add_ddu_input_g else 0) + (3 if add_dddu_input_g else 0) + (1 if add_p_prev_input_g else 0) + (1 if add_dp_prev_input_g else 0) + (1 if add_ddp_prev_input_g else 0) + 1 + int(add_div_ddu_input_g) + int(add_div_du_input_g) + int(add_div_u_input_g)
		actual_channels = x_array.shape[-1]
		if verbose_g:
			print(f"[py_func DEBUG] x_array shape: {x_array.shape}")
			print(f"[py_func DEBUG] Expected channels: {expected_channels}, Actual channels: {actual_channels}")
			print(f"[py_func DEBUG] Input flags - add_U: {add_U_input_g}, add_dU: {add_dU_input_g}, add_ddu: {add_ddu_input_g}, add_dddu: {add_dddu_input_g}, add_p_prev: {add_p_prev_input_g}, use_dp_prev: {add_dp_prev_input_g}, add_ddp_prev: {add_ddp_prev_input_g}")
		if actual_channels != expected_channels:
			raise ValueError(f"Channel mismatch in x_array: expected {expected_channels}, got {actual_channels}")

		if verbose_g:
			print(f"Data extraction loop took: {time.time()-t0} s")

		t0 = time.time()
		N = x_array.shape[0]

		if use_feature_decomposition_g:
			A = in_factors_T[1]
			B = in_factors_T[2]
			C = in_factors_T[3]
			D = in_factors_T[4]

			input_core = np.einsum("nijkf,ai,bj,ck,df->nabcd", x_array, A, B, C, D, optimize=True)
			input_transformed = input_core.reshape(N, -1)
			x_input = (input_transformed - mean_in) / std_in
		else:
			# No Tucker: pass raw normalized blocks directly to the 3D CNN
			x_input = (x_array - mean_in) / std_in

		if verbose_g:
			print(f"{'Tucker transformation' if use_feature_decomposition_g else 'Input standardization'} : {time.time()-t0} s")

		t0 = time.time()
		
		if effective_model_arch_g.lower() == 'cnn_two_heads':
			outputs = model(x_input, training=False)

			res_concat = np.array(outputs["p_total"])
			if inspect_results_g:
				if "p_smooth" in outputs:
					p_smooth_raw = np.array(outputs["p_smooth"])
				else:
					p_smooth_raw = None
				if "p_local" in outputs:
					p_local_raw  = np.array(outputs["p_local"])
				else:
					p_local_raw = None
			else:
				p_smooth_raw = p_local_raw = None
		else:
			res_concat = np.array(model(x_input, training=False))
			p_smooth_raw = p_local_raw = None

		if verbose_g:
			print(f"Model prediction time : {time.time()-t0} s")

		t0 = time.time()

		res_concat = (res_concat * std_out) + mean_out
		if p_smooth_raw is not None:
			p_smooth_raw = (p_smooth_raw * std_out) + mean_out
			p_local_raw  = (p_local_raw  * std_out) + mean_out

		if use_feature_decomposition_g:
			# CHANGED: avoid extra copy in reshape
			core = res_concat.reshape(input_core[..., 0].shape)
			if core.dtype != np.float64:
				core = core.astype(np.float64, copy=False)

			U1 = out_factors_c[1]
			U2 = out_factors_c[2]
			U3 = out_factors_c[3]

			res_concat = np.einsum("nabc,ia,jb,kc->nijk", core, U1, U2, U3, optimize=True)

			if p_smooth_raw is not None:
				core_s = p_smooth_raw.reshape(input_core[..., 0].shape).astype(np.float64, copy=False)
				p_smooth_raw = np.einsum("nabc,ia,jb,kc->nijk", core_s, U1, U2, U3, optimize=True)
				core_l = p_local_raw.reshape(input_core[..., 0].shape).astype(np.float64, copy=False)
				p_local_raw = np.einsum("nabc,ia,jb,kc->nijk", core_l, U1, U2, U3, optimize=True)

			if verbose_g:
				print(f"Tucker inverse transform : {time.time()-t0} s")
		else:
			# res_concat already has shape (N, bsz, bsy, bsx) — no inverse transform needed
			if res_concat.dtype != np.float64:
				res_concat = res_concat.astype(np.float64, copy=False)
			if p_smooth_raw is not None:
				if p_smooth_raw.dtype != np.float64:
					p_smooth_raw = p_smooth_raw.astype(np.float64, copy=False)
				if p_local_raw.dtype != np.float64:
					p_local_raw = p_local_raw.astype(np.float64, copy=False)
			if verbose_g:
				print(f"Output destandardization (no Tucker inverse) : {time.time()-t0} s")

		t0 = time.time()

		# CHANGED: in-place multiply to avoid copy
		res_concat *= max_abs_delta_delta_p * pow(U_max_norm, 2.0)
		if p_smooth_raw is not None:
			p_smooth_raw *= max_abs_delta_delta_p * pow(U_max_norm, 2.0)
			p_local_raw  *= max_abs_delta_delta_p * pow(U_max_norm, 2.0)

		number_of_nans = np.isnan(res_concat).sum()
		if verbose_g:
			print(f"Number of NaNs in res_concat before assembling: {number_of_nans}/{res_concat.size}")
		assert number_of_nans == 0, "NaN values found in res_concat before assembling."

		if verbose_g:
			print(f"Processing before assembly : {time.time()-t0} s")

		t0 = time.time()

		Ref_BC = 0
		global _inspect_call_count
		if inspect_results_g:
			_inspect_call_count += 1
			
		# SPETIAL CASE: block representing the full simulation
		change_in_deltap = assemble_prediction(
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
				delta_U_change_grid,
				delta_delta_p_prev_grid,
				apply_deltaU_change_wgt=False,
				filter_tuple = filter_tuple_g,
				filter_tuple_deltaU = (5,5,5),
				plot_results=inspect_results_g,
				plot_call_count=_inspect_call_count,
				plot_output_dir=inspect_output_dir_g,
			)

		if verbose_g:
			print(f"Assembly algorithm took: {time.time()-t0} s")

		# Assemble and plot p_smooth / p_local grids (two-head model, inspect mode only)
		if inspect_results_g and p_smooth_raw is not None:
			_assemble_kwargs = dict(
				indices_list=indices_list,
				n_x=n_x, n_y=n_y, n_z=n_z,
				overlap=overlap,
				shape=block_size,
				Ref_BC=Ref_BC,
				x_array=x_array,
				apply_filter=apply_filter_g,
				shape_x=grid_shape_x, shape_y=grid_shape_y, shape_z=grid_shape_z,
				delta_U_change_grid=delta_U_change_grid,
				delta_delta_p_prev_grid=delta_delta_p_prev_grid,
				apply_deltaU_change_wgt=False,
				filter_tuple=filter_tuple_g,
				filter_tuple_deltaU=(5, 5, 5),
				plot_results=False,
				plot_call_count=_inspect_call_count,
				plot_output_dir=inspect_output_dir_g,
			)
			p_smooth_assembled = assemble_prediction(p_smooth_raw, **_assemble_kwargs)
			p_local_assembled  = assemble_prediction(p_local_raw,  **_assemble_kwargs)
			_plot_two_head_decomposition_multiZ(
				p_smooth_assembled,
				p_local_assembled,
				_inspect_call_count,
				inspect_output_dir_g,
			)

		t0 = time.time()

		# CHANGED: use cached indices arrays for faster indexing
		change_in_deltap = change_in_deltap[indices_i, indices_j, indices_k]

		number_of_nans = np.isnan(change_in_deltap).sum()
		if verbose_g:
			print(f"Max and min of change_in_deltap before filtering: {np.nanmax(change_in_deltap)}, {np.nanmin(change_in_deltap)}")		
			print(f"Number of NaNs in change_in_deltap before filtering: {number_of_nans}/{change_in_deltap.size}")
			assert number_of_nans == 0, "NaN values found in change_in_deltap before filtering."
			print(f"Flattening array to send to OF and checking NANs took: {time.time()-t0} s")
		
		t0 = time.time()

		delta_delta_p = interpolate_fill_njit(change_in_deltap, vert_NPtoOF_array, weights_NPtoOF_array)
		
		#p = deltaP_prev + change_in_deltap

		if verbose_g:
			print(f"Final Interpolation took: {time.time()-t0} s")

		# CHANGED: vectorized split using np.split (faster than manual loop)
		if len(len_rankwise) != nprocs:
			raise ValueError(f"len_rankwise ({len(len_rankwise)}) does not match number of ranks ({nprocs})")
		if sum(len_rankwise) != len(delta_delta_p):
			raise ValueError(f"Sum of len_rankwise ({sum(len_rankwise)}) does not match length of delta_delta_p ({len(delta_delta_p)})")

		delta_delta_p_rankwise = np.split(delta_delta_p, np.cumsum(len_rankwise)[:-1])

		if verbose_g:
			print(f"The whole python function took : {time.time()-t0_py_func} s")

	else:
		delta_delta_p_rankwise = None

	for delta_delta_p_rank_i in delta_delta_p_rankwise:
		if np.any(np.isnan(delta_delta_p_rank_i)):
			nan_count = np.sum(np.isnan(delta_delta_p_rank_i))
			if verbose_g:
				print(f"Number of NaN values in delta_delta_p_rankwise: {nan_count}")
			raise ValueError("Warning: NaN values detected in delta_delta_p_rankwise before scattering.")

	# This scatters the value to each worker
	delta_delta_p = comm.scatter(delta_delta_p_rankwise, root=0)

	if verbose_g:
		if np.any(np.isnan(delta_delta_p)):
			print(f"Warning: NaN values detected in delta_delta_p at rank {rank} after scattering.")

	if verbose_g:
		print(f"Process {rank} received object with shape {delta_delta_p.shape}")

	# Enforce float64 dtype and check for NaNs/Infs before returning to C++
	delta_delta_p = np.asarray(delta_delta_p, dtype=np.float64)
	if not np.all(np.isfinite(delta_delta_p)):
		n_nan = np.isnan(delta_delta_p).sum()
		n_inf = np.isinf(delta_delta_p).sum()
		raise ValueError(f"Output array contains {n_nan} NaNs and {n_inf} Infs before returning to C++.")
	
	return delta_delta_p * 0.5

if __name__ == '__main__':
    print('This is the Python module for DLPoissonFOam')