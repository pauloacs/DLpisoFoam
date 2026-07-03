###################################################################################################
###################################################################################################
########################## STILL WORKING ON MAKING THIS WORK ######################################
###################################################################################################
###################################################################################################

import os
from unittest import result
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
import matplotlib
matplotlib.use("Agg")  # Force headless backend: embedded MPI Python has no display; GUI backends segfault
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import tensorly as tl

from pressure_SM_delta_delta_shift._3D.CFD_usable.utils import memory
from pressure_SM_delta_delta_shift._3D.train_and_eval.utils.data_processing import interpolate_fill_njit, interp_weights, create_uniform_grid, _unpack_grid_res
from pressure_SM_delta_delta_shift._3D.train_and_eval.utils.domain_geometry import domain_dist
from pressure_SM_delta_delta_shift._3D.train_and_eval.utils.model_utils import define_model_arch
from pressure_SM_delta_delta_shift._3D.train_and_eval.utils import data_processing as utils_data

from pressure_SM_delta_delta_shift._3D.train_and_eval.assembly import assemble_prediction
from pressure_SM_delta_delta_shift._3D.train_and_eval.neural_networks import (
    MLP, SimpleCNN3D, Simple_multi_layer_3D, FNO3d, GNN, MLP_Mixer_3D, 
    UNet3D, UNet3D_deep, UNet3D_attention, SymmetricPadding3D, SimpleCNN3D_two_heads, SimpleCNN3D_two_heads_smooth, SimpleCNN3D_multi_out, SimpleCNN3D_multi_out_divU
)
from pressure_SM_delta_delta_shift._3D.train_and_eval.neural_networks_shifter import (
	SimpleCNN3D_ddp_shifter,
	SimpleCNN3D_ddp_shifter_lightweight,
	SimpleCNN3D_ddp_shifter_velocity,
)


_inspect_call_count = 0
_py_func_call_count = 0  # Track py_func calls for oracle data indexing


def _compute_grad_dpprev_channel_idxs():
	"""Compute gradDpPrev channel indices in x_array/std-normalized x_input."""
	if not add_gradDpPrev_input_g:
		return None

	ch_idx = 0
	if add_U_input_g:
		ch_idx += 3
	if add_dUStar_input_g:
		ch_idx += 3
	if add_ddUStar_input_g:
		ch_idx += 3
	if add_ddUStarDiff_input_g:
		ch_idx += 3
	if add_dUCorrPrev_input_g:
		ch_idx += 3
	if add_ddUCorrPrev_input_g:
		ch_idx += 3
	if add_pPrev_input_g:
		ch_idx += 1
	if add_dpPrev_input_g:
		ch_idx += 1
	if add_ddpPrev_input_g:
		ch_idx += 1
	# gradDpPrev channels are placed before optional scalar dpPrev-derived inputs
	# in the raw block layout: [..., ddpPrev, gradDpPrev(x,y,z), laplacian, uDotGrad, |grad|, ...]
	return (ch_idx, ch_idx + 1, ch_idx + 2)


def _compute_u_channel_idxs():
	"""Compute U channel indices in x_array/std-normalized x_input."""
	if not add_U_input_g:
		return None
	return (0, 1, 2)


def _compute_u_dot_grad_dpprev_channel_idx():
	"""Compute U.dot.grad(dP_prev) channel index in x_array/std-normalized x_input."""
	if not add_uDotGradDpPrev_input_g:
		return None

	ch_idx = 0
	if add_U_input_g:
		ch_idx += 3
	if add_dUStar_input_g:
		ch_idx += 3
	if add_ddUStar_input_g:
		ch_idx += 3
	if add_ddUStarDiff_input_g:
		ch_idx += 3
	if add_dUCorrPrev_input_g:
		ch_idx += 3
	if add_ddUCorrPrev_input_g:
		ch_idx += 3
	if add_pPrev_input_g:
		ch_idx += 1
	if add_dpPrev_input_g:
		ch_idx += 1
	if add_ddpPrev_input_g:
		ch_idx += 1
	if add_gradDpPrev_input_g:
		ch_idx += 3
	if add_laplacian_dpPrev_input_g:
		ch_idx += 1
	return ch_idx


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
		plt.savefig(fname, dpMLi=100, bbox_inches="tight")
		plt.close(fig)
		print(f"[two_heads] Saved: {fname}")


def _plot_ddp_pred_vs_prev_z_slices(
	ddp_pred_grid,
	ddp_prev_grid,
	call_count,
	output_dir,
	max_abs_ddp=1.0,
	U_max_norm=1.0,
	z_slices=[0, 20, 40, 60, 80, 95],
):
	"""
	Compare the assembled predicted ddP against the previous-step ddP on
	several z-slices. Mirrors the train.py ddP comparison so that the runtime
	prediction can be verified visually in the same way.

	Two figures are produced:
	  * physical space  (values as handed to the CFD solver)
	  * common space    (÷ U_max_norm**2, the normalized space train.py plots)

	ddp_pred_grid, ddp_prev_grid : (grid_z, grid_y, grid_x) physical arrays.
	"""
	import matplotlib.pyplot as plt

	os.makedirs(output_dir, exist_ok=True)

	z_len = ddp_pred_grid.shape[0]
	indices = [int((s / 100) * z_len) for s in z_slices]
	n_panels = len(indices)

	u2 = float(U_max_norm) ** 2 if U_max_norm else 1.0

	for space_label, scale, fname_key in [
		("physical", 1.0, "ddp_pred_vs_prev_phys"),
		("common (/U_max_norm^2)", 1.0 / (u2 + 1e-30), "ddp_pred_vs_prev_common"),
	]:
		pred = ddp_pred_grid * scale
		prev = ddp_prev_grid * scale

		fig, axes = plt.subplots(n_panels, 2, figsize=(10, 2.5 * n_panels), squeeze=False)

		for panel_idx, z_idx in enumerate(indices):
			pred_slice = pred[z_idx, :, :]
			prev_slice = prev[z_idx, :, :]
			# Shared symmetric color scale per row for a fair magnitude comparison
			vmax = float(np.nanmax(np.abs(np.concatenate([pred_slice.ravel(), prev_slice.ravel()]))))
			if not np.isfinite(vmax) or vmax <= 0.0:
				vmax = 1e-8

			for col, (data, col_label) in enumerate(
				[(pred_slice, "ddP_pred"), (prev_slice, "ddP_prev")]
			):
				im = axes[panel_idx, col].imshow(
					data, origin="lower", aspect="auto",
					cmap="RdBu_r", vmin=-vmax, vmax=vmax,
				)
				axes[panel_idx, col].set_title(
					f"{col_label} z={z_idx}", fontsize=12, fontweight="bold"
				)
				axes[panel_idx, col].axis("off")
				plt.colorbar(im, ax=axes[panel_idx, col], fraction=0.046, pad=0.04)

		# Annotate global magnitude ratio for quick sanity checking
		mean_pred = float(np.mean(np.abs(pred)))
		mean_prev = float(np.mean(np.abs(prev)))
		ratio = mean_pred / (mean_prev + 1e-30)
		fig.suptitle(
			f"ddP pred vs prev [{space_label}] - Call {call_count}\n"
			f"mean|pred|={mean_pred:.3e}  mean|prev|={mean_prev:.3e}  pred/prev={ratio:.4f}",
			fontsize=13, fontweight="bold",
		)
		plt.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.05, hspace=0.35, wspace=0.15)
		fname = os.path.join(output_dir, f"{fname_key}_{call_count:05d}.png")
		plt.savefig(fname, dpi=90, bbox_inches="tight")
		plt.close(fig)
		print(f"[ddp_pred_vs_prev] Saved: {fname}")


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
	add_ddUStar_input=True,
	add_ddUStarDiff_input=True,
	add_U_input=False,
	add_dUStar_input=False,
	add_dpPrev_input=False,
	add_pPrev_input=False,
	add_ddpPrev_input=False,
	add_gradDpPrev_input=False,
	add_laplacian_dpPrev_input=False,
	add_uDotGradDpPrev_input=False,
	add_gradDpPrevMag_input=False,
	include_rAU_input=False,
	include_HbyA_input=False,
	include_divHbyA_input=False,
	include_dHbyA_input=False,
	include_dDivHbyA_input=False,
	add_rAUGradDpPrev_input=False,
	add_divRAUGradDpPrev_input=False,
	add_pressureEqResidualp_input=False,
	add_rAUGradpPrev_input=False,
	add_divRAUGradpPrev_input=False,
	add_divDDUStar_input=False,
	add_divDUStar_input=False,
	add_divUStar_input=False,
	add_dUCorrPrev_input=False,
	add_ddUCorrPrev_input=False,
	predict_ddUCorr=False,
	output_weight_factor=1.0,
	oracle_mode=False,
	oracle_model_with_interp=False,
	oracle_data_folder='ML_data',
	enforce_zero_mean_pressure=True,
	add_distance_to_outlet_input=False,
	add_grad_sdf_input=False,
	add_UdotNwall_input=False,
	clip_UdotNwall_to_inflow=False,
	use_s_roi_penalty=False,
	compression_clip_percentile=75.0,
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
	global compression_clip_percentile_g
	compression_clip_percentile_g = float(compression_clip_percentile)

	if model_arch.lower() in ['cnn_shifter', 'cnn_shifter_lightweight', 'simplecnn3d_ddp_shifter', 'simplecnn3d_ddp_shifter_lightweight', 'cnn_shifter_velocity', 'simplecnn3d_ddp_shifter_velocity'] and use_feature_decomposition:
		raise ValueError('[load_tucker_and_NN] cnn_shifter requires use_feature_decomposition=False.')

	# Store these so init_func can rebuild the CNN with the correct block shape
	global dropout_rate_g, regularization_g, n_layers_g, width_g, weights_fn_g
	dropout_rate_g = dropout_rate
	regularization_g = regularization
	weights_fn_g = weights_fn
	n_layers_g, width_g = define_model_arch(model_arch)
	global effective_model_arch_g
	_FLAT_MODELS_EARLY = ('mlp_small', 'mlp_big', 'mlp_small_unet', 'mlp_huge', 'mlp_huger', 'conv1d', 'mlp_attention')
	effective_model_arch_g = 'cnn' if (not use_feature_decomposition and model_arch.lower() in _FLAT_MODELS_EARLY) else model_arch
	# When predicting ddU, redirect cnn_two_heads to the multi-output variant
	if predict_ddUCorr and effective_model_arch_g.lower() == 'cnn_two_heads':
		print('[load_tucker_and_NN] predict_ddUCorr=True with cnn_two_heads - using cnn_multi_out variant.')
		effective_model_arch_g = 'cnn_multi_out'
	if effective_model_arch_g.lower() == 'cnn_multi_out_divu' and use_feature_decomposition:
		print('[load_tucker_and_NN] cnn_multi_out_divu requires use_feature_decomposition=False. Falling back to cnn_multi_out.')
		effective_model_arch_g = 'cnn_multi_out'

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


		global max_abs_U_x, max_abs_U_y, max_abs_U_z, max_abs_dU_x, max_abs_dU_y, max_abs_dU_z, max_abs_ddU_x, max_abs_ddU_y, max_abs_ddU_z, max_abs_dddU_x, max_abs_dddU_y, max_abs_dddU_z, max_abs_dist, max_abs_ddp, max_abs_dpPrev, max_abs_ddpPrev, max_abs_gradDpPrev_x, max_abs_gradDpPrev_y, max_abs_gradDpPrev_z, max_abs_laplacian_dpPrev, max_abs_uDotGradDpPrev, max_abs_gradDpPrevMag, max_abs_p_prev, max_abs_div_ddu, max_abs_div_du, max_abs_div_u, max_abs_delta_delta_U_x, max_abs_delta_delta_U_y, max_abs_delta_delta_U_z
		global max_abs_rAU, max_abs_HbyA_x, max_abs_HbyA_y, max_abs_HbyA_z, max_abs_divHbyA
		global max_abs_dHbyA_x, max_abs_dHbyA_y, max_abs_dHbyA_z, max_abs_dDivHbyA
		global max_abs_rAUGradDpPrev_x, max_abs_rAUGradDpPrev_y, max_abs_rAUGradDpPrev_z, max_abs_divRAUGradDpPrev, max_abs_pressureEqResidualp, max_abs_rAUGradpPrev_x, max_abs_rAUGradpPrev_y, max_abs_rAUGradpPrev_z, max_abs_divRAUGradpPrev
		global max_abs_dUCorrPrev_x, max_abs_dUCorrPrev_y, max_abs_dUCorrPrev_z, max_abs_ddUCorrPrev_x, max_abs_ddUCorrPrev_y, max_abs_ddUCorrPrev_z
		global max_abs_dist_to_outlet, max_abs_grad_sdf_x, max_abs_grad_sdf_y, max_abs_grad_sdf_z, max_abs_UdotNwall
		global add_U_input_g, add_dUStar_input_g, add_ddUStar_input_g, add_ddUStarDiff_input_g, add_dpPrev_input_g, add_pPrev_input_g, add_ddpPrev_input_g, add_gradDpPrev_input_g, add_laplacian_dpPrev_input_g, add_uDotGradDpPrev_input_g, add_gradDpPrevMag_input_g, add_rAUGradDpPrev_input_g, add_divRAUGradDpPrev_input_g, add_pressureEqResidualp_input_g, add_rAUGradpPrev_input_g, add_divRAUGradpPrev_input_g, add_divDDUStar_input_g, add_divDUStar_input_g, add_divUStar_input_g, add_dUCorrPrev_input_g, add_ddUCorrPrev_input_g
		global include_rAU_input_g, include_HbyA_input_g, include_divHbyA_input_g
		global include_dHbyA_input_g, include_dDivHbyA_input_g
		global predict_ddUCorr_g, output_weight_factor_g, oracle_mode_g, oracle_data_folder_g, oracle_call_counter_g, oracle_ddp_cache_g, oracle_model_with_interp_g
		add_U_input_g = add_U_input
		add_dUStar_input_g = add_dUStar_input
		add_ddUStar_input_g = add_ddUStar_input
		add_ddUStarDiff_input_g = add_ddUStarDiff_input
		add_dpPrev_input_g = add_dpPrev_input
		add_pPrev_input_g = add_pPrev_input
		add_ddpPrev_input_g = add_ddpPrev_input
		add_gradDpPrev_input_g = add_gradDpPrev_input
		add_laplacian_dpPrev_input_g = add_laplacian_dpPrev_input
		add_uDotGradDpPrev_input_g = add_uDotGradDpPrev_input
		add_gradDpPrevMag_input_g = add_gradDpPrevMag_input
		include_rAU_input_g = include_rAU_input
		include_HbyA_input_g = include_HbyA_input
		include_divHbyA_input_g = include_divHbyA_input
		include_dHbyA_input_g = include_dHbyA_input
		include_dDivHbyA_input_g = include_dDivHbyA_input
		add_rAUGradDpPrev_input_g = add_rAUGradDpPrev_input
		add_divRAUGradDpPrev_input_g = add_divRAUGradDpPrev_input
		add_pressureEqResidualp_input_g = add_pressureEqResidualp_input
		add_rAUGradpPrev_input_g = add_rAUGradpPrev_input
		add_divRAUGradpPrev_input_g = add_divRAUGradpPrev_input
		add_divDDUStar_input_g = add_divDDUStar_input
		add_divDUStar_input_g = add_divDUStar_input
		add_divUStar_input_g = add_divUStar_input
		add_dUCorrPrev_input_g = add_dUCorrPrev_input
		add_ddUCorrPrev_input_g = add_ddUCorrPrev_input

		# Legacy divergence channels are no longer exported separately by the C++ solver.
		# Only divUFirstPred exists in raw column 15 and is mapped to add_divUStar_input.
		if add_divDDUStar_input or add_divDUStar_input:
			raise ValueError(
				"add_divDDUStar_input/add_divDUStar_input are unsupported with the current solver raw layout. "
				"Only divUFirstPred is exported (use add_divUStar_input)."
			)
		predict_ddUCorr_g = predict_ddUCorr
		output_weight_factor_g = output_weight_factor
		oracle_mode_g = oracle_mode
		oracle_model_with_interp_g = oracle_model_with_interp
		oracle_data_folder_g = oracle_data_folder
		oracle_ddp_cache_g = []  # Array for oracle ddp values, populated in init_func
		oracle_ddu_cache_g = []  # Array for oracle delta_delta_U values, populated in init_func
		global enforce_zero_mean_pressure_g
		enforce_zero_mean_pressure_g = enforce_zero_mean_pressure
		global add_distance_to_outlet_input_g, add_grad_sdf_input_g, add_UdotNwall_input_g, clip_UdotNwall_to_inflow_g
		add_distance_to_outlet_input_g = add_distance_to_outlet_input
		add_grad_sdf_input_g = add_grad_sdf_input
		add_UdotNwall_input_g = add_UdotNwall_input
		clip_UdotNwall_to_inflow_g = clip_UdotNwall_to_inflow
		global gradDpPrev_input_ch_idxs_g
		gradDpPrev_input_ch_idxs_g = _compute_grad_dpprev_channel_idxs()
		global U_input_ch_idxs_g
		U_input_ch_idxs_g = _compute_u_channel_idxs()
		global uDotGradDpPrev_input_ch_idx_g
		uDotGradDpPrev_input_ch_idx_g = _compute_u_dot_grad_dpprev_channel_idx()

		# Expected number of input channels in gridded tensors for the current flag set.
		expected_input_ch = (
			(3 if add_U_input else 0)
			+ (3 if add_dUStar_input else 0)
			+ (3 if add_ddUStar_input else 0)
			+ (3 if add_ddUStarDiff_input else 0)
			+ (3 if add_dUCorrPrev_input else 0)
			+ (3 if add_ddUCorrPrev_input else 0)
			+ (1 if add_pPrev_input else 0)
			+ (1 if add_dpPrev_input else 0)
			+ (1 if add_ddpPrev_input else 0)
			+ (3 if add_gradDpPrev_input else 0)
			+ (1 if add_laplacian_dpPrev_input else 0)
			+ (1 if add_uDotGradDpPrev_input else 0)
			+ (1 if add_gradDpPrevMag_input else 0)
			+ (1 if include_rAU_input else 0)
			+ (3 if include_HbyA_input else 0)
			+ (1 if include_divHbyA_input else 0)
			+ (3 if include_dHbyA_input else 0)
			+ (1 if include_dDivHbyA_input else 0)
			+ (3 if add_rAUGradDpPrev_input else 0)
			+ (1 if add_divRAUGradDpPrev_input else 0)
			+ (1 if add_pressureEqResidualp_input else 0)
			+ (3 if add_rAUGradpPrev_input else 0)
			+ (1 if add_divRAUGradpPrev_input else 0)
			+ (1 if add_divDDUStar_input else 0)
			+ (1 if add_divDUStar_input else 0)
			+ (1 if add_divUStar_input else 0)
			+ (1 if add_distance_to_outlet_input else 0)
			+ (3 if add_grad_sdf_input else 0)
			+ (1 if add_UdotNwall_input else 0)
			+ 1  # sdf
		)
		expected_maxs_len = expected_input_ch + 1 + (3 if predict_ddUCorr else 0)
		if len(maxs) < expected_maxs_len:
			raise ValueError(
				f"maxs length mismatch for current input flags: got {len(maxs)}, expected at least {expected_maxs_len}. "
				"ML_data/maxs appears to come from a different enabled-input configuration. "
				"Regenerate ML_data (train_init) with the current python_module flags."
			)
		
		# Parse maxs based on flags: [U if add_U] [dU if add_dU] [ddU if add_ddu] dddU dist [p_prev if add_p_prev] [dpPrev if add_dpPrev] [ddpPrev if add_ddpPrev] ddp [ddU_x/y/z if predict_ddUCorr]
		ch_idx = 0
		if add_U_input:
			max_abs_U_x, max_abs_U_y, max_abs_U_z = maxs[ch_idx:ch_idx+3]
			ch_idx += 3
		else:
			max_abs_U_x = max_abs_U_y = max_abs_U_z = 1.0  # unused

		if add_dUStar_input:
			max_abs_dU_x, max_abs_dU_y, max_abs_dU_z = maxs[ch_idx:ch_idx+3]
			ch_idx += 3
		else:
			max_abs_dU_x = max_abs_dU_y = max_abs_dU_z = 1.0  # unused
		
		if add_ddUStar_input:
			max_abs_ddU_x, max_abs_ddU_y, max_abs_ddU_z = maxs[ch_idx:ch_idx+3]
			ch_idx += 3
		else:
			max_abs_ddU_x = max_abs_ddU_y = max_abs_ddU_z = 1.0  # unused
		
		if add_ddUStarDiff_input:
			max_abs_dddU_x, max_abs_dddU_y, max_abs_dddU_z = maxs[ch_idx:ch_idx+3]
			ch_idx += 3
		else:
			max_abs_dddU_x = max_abs_dddU_y = max_abs_dddU_z = 1.0  # unused
		if add_dUCorrPrev_input:
			max_abs_dUCorrPrev_x, max_abs_dUCorrPrev_y, max_abs_dUCorrPrev_z = maxs[ch_idx:ch_idx+3]
			ch_idx += 3
		else:
			max_abs_dUCorrPrev_x = max_abs_dUCorrPrev_y = max_abs_dUCorrPrev_z = 1.0  # unused
		if add_ddUCorrPrev_input:
			max_abs_ddUCorrPrev_x, max_abs_ddUCorrPrev_y, max_abs_ddUCorrPrev_z = maxs[ch_idx:ch_idx+3]
			ch_idx += 3
		else:
			max_abs_ddUCorrPrev_x = max_abs_ddUCorrPrev_y = max_abs_ddUCorrPrev_z = 1.0  # unused
		max_abs_dist = maxs[ch_idx]
		ch_idx += 1
		if add_pPrev_input:
			max_abs_p_prev = maxs[ch_idx]
			ch_idx += 1
		else:
			max_abs_p_prev = 1.0  # unused
		if add_dpPrev_input:
			max_abs_dpPrev = maxs[ch_idx]
			ch_idx += 1
		else:
			max_abs_dpPrev = 1.0  # unused
		if add_ddpPrev_input:
			max_abs_ddpPrev = maxs[ch_idx]
			ch_idx += 1
		else:
			max_abs_ddpPrev = 1.0  # unused
		if add_gradDpPrev_input:
			max_abs_gradDpPrev_x, max_abs_gradDpPrev_y, max_abs_gradDpPrev_z = maxs[ch_idx:ch_idx+3]
			ch_idx += 3
		else:
			max_abs_gradDpPrev_x = max_abs_gradDpPrev_y = max_abs_gradDpPrev_z = 1.0  # unused
		if add_laplacian_dpPrev_input:
			max_abs_laplacian_dpPrev = maxs[ch_idx]
			ch_idx += 1
		else:
			max_abs_laplacian_dpPrev = 1.0
		if add_uDotGradDpPrev_input:
			max_abs_uDotGradDpPrev = maxs[ch_idx]
			ch_idx += 1
		else:
			max_abs_uDotGradDpPrev = 1.0
		if add_gradDpPrevMag_input:
			max_abs_gradDpPrevMag = maxs[ch_idx]
			ch_idx += 1
		else:
			max_abs_gradDpPrevMag = 1.0
		if include_rAU_input:
			max_abs_rAU = maxs[ch_idx]
			ch_idx += 1
		else:
			max_abs_rAU = 1.0
		if include_HbyA_input:
			max_abs_HbyA_x, max_abs_HbyA_y, max_abs_HbyA_z = maxs[ch_idx:ch_idx+3]
			ch_idx += 3
		else:
			max_abs_HbyA_x = max_abs_HbyA_y = max_abs_HbyA_z = 1.0
		if include_divHbyA_input:
			max_abs_divHbyA = maxs[ch_idx]
			ch_idx += 1
		else:
			max_abs_divHbyA = 1.0
		if include_dHbyA_input:
			max_abs_dHbyA_x, max_abs_dHbyA_y, max_abs_dHbyA_z = maxs[ch_idx:ch_idx+3]
			ch_idx += 3
		else:
			max_abs_dHbyA_x = max_abs_dHbyA_y = max_abs_dHbyA_z = 1.0
		if include_dDivHbyA_input:
			max_abs_dDivHbyA = maxs[ch_idx]
			ch_idx += 1
		else:
			max_abs_dDivHbyA = 1.0
		if add_rAUGradDpPrev_input:
			max_abs_rAUGradDpPrev_x, max_abs_rAUGradDpPrev_y, max_abs_rAUGradDpPrev_z = maxs[ch_idx:ch_idx+3]
			ch_idx += 3
		else:
			max_abs_rAUGradDpPrev_x = max_abs_rAUGradDpPrev_y = max_abs_rAUGradDpPrev_z = 1.0
		if add_divRAUGradDpPrev_input:
			max_abs_divRAUGradDpPrev = maxs[ch_idx]
			ch_idx += 1
		else:
			max_abs_divRAUGradDpPrev = 1.0
		if add_pressureEqResidualp_input:
			max_abs_pressureEqResidualp = maxs[ch_idx]
			ch_idx += 1
		else:
			max_abs_pressureEqResidualp = 1.0
		if add_rAUGradpPrev_input:
			max_abs_rAUGradpPrev_x, max_abs_rAUGradpPrev_y, max_abs_rAUGradpPrev_z = maxs[ch_idx:ch_idx+3]
			ch_idx += 3
		else:
			max_abs_rAUGradpPrev_x = max_abs_rAUGradpPrev_y = max_abs_rAUGradpPrev_z = 1.0
		if add_divRAUGradpPrev_input:
			max_abs_divRAUGradpPrev = maxs[ch_idx]
			ch_idx += 1
		else:
			max_abs_divRAUGradpPrev = 1.0
		if add_divDDUStar_input:
			max_abs_div_ddu = maxs[ch_idx]
			ch_idx += 1
		else:
			max_abs_div_ddu = 1.0  # unused
		if add_divDUStar_input:
			max_abs_div_du = maxs[ch_idx]
			ch_idx += 1
		else:
			max_abs_div_du = 1.0  # unused
		if add_divUStar_input:
			max_abs_div_u = maxs[ch_idx]
			ch_idx += 1
		else:
			max_abs_div_u = 1.0  # unused
		if add_distance_to_outlet_input:
			max_abs_dist_to_outlet = maxs[ch_idx]
			ch_idx += 1
		else:
			max_abs_dist_to_outlet = 1.0  # unused
		if add_grad_sdf_input:
			max_abs_grad_sdf_x, max_abs_grad_sdf_y, max_abs_grad_sdf_z = maxs[ch_idx:ch_idx+3]
			ch_idx += 3
		else:
			max_abs_grad_sdf_x = max_abs_grad_sdf_y = max_abs_grad_sdf_z = 1.0  # unused
		
		if add_UdotNwall_input:
			max_abs_UdotNwall = maxs[ch_idx]
			ch_idx += 1
		else:
			max_abs_UdotNwall = 1.0  # unused
		max_abs_ddp = maxs[ch_idx]
		ch_idx += 1
		if predict_ddUCorr:
			if ch_idx + 3 <= len(maxs):
				max_abs_delta_delta_U_x, max_abs_delta_delta_U_y, max_abs_delta_delta_U_z = maxs[ch_idx:ch_idx+3]
				ch_idx += 3
			else:
				print('[load_tucker_and_NN] WARNING: maxs file does not contain predict_ddUCorr entries '
				      f'(expected at least {ch_idx+3} values, got {len(maxs)}). '
				      'Using 1.0 as fallback — re-run train_init with predict_ddUCorr_output=True.')
				max_abs_delta_delta_U_x = max_abs_delta_delta_U_y = max_abs_delta_delta_U_z = 1.0
		else:
			max_abs_delta_delta_U_x = max_abs_delta_delta_U_y = max_abs_delta_delta_U_z = 1.0  # unused
		
		# Loading values for standardization
		data = np.load(std_vals_fn)
		global mean_in, std_in, mean_out, std_out
		mean_in = data['mean_in']
		std_in = data['std_in']
		mean_out = data['mean_out']
		std_out = data['std_out']
		if np.ravel(mean_in).size != expected_input_ch or np.ravel(std_in).size != expected_input_ch:
			raise ValueError(
				f"mean/std input size mismatch for current flags: mean_in={np.ravel(mean_in).size}, "
				f"std_in={np.ravel(std_in).size}, expected={expected_input_ch}. "
				"ML_data/mean_std.npz appears to come from a different enabled-input configuration. "
				"Regenerate ML_data (train_init) with the current python_module flags."
			)

		# Auto-calculate last_tucker_rank if not provided
		# Base is 4; each input flag adds 3, add_dpPrev_input adds 1: [U if add_U_input] [ddU if add_ddUStar_input] dddU sdf dpML [dpPrev if add_dpPrev_input] -> mapped through Tucker
		# Actually for CNN (use_feature_decomposition=False), it's: [U if add_U_input] [ddU if add_ddUStar_input] dddU [+dpPrev if add_dpPrev_input] -> input channels
		# For Tucker decomposition (use_feature_decomposition=True), it uses the last Tucker factor dimension
		last_tucker_rank = 1  # base: sdf only
		if add_U_input:
			last_tucker_rank += 3
		if add_dUStar_input:
			last_tucker_rank += 3
		if add_ddUStar_input:
			last_tucker_rank += 3
		if add_ddUStarDiff_input:
			last_tucker_rank += 3
		if add_dUCorrPrev_input:
			last_tucker_rank += 3
		if add_ddUCorrPrev_input:
			last_tucker_rank += 3
		if add_pPrev_input:
			last_tucker_rank += 1
		if add_dpPrev_input:
			last_tucker_rank += 1
		if add_ddpPrev_input:
			last_tucker_rank += 1
		if add_gradDpPrev_input:
			last_tucker_rank += 3
		if add_laplacian_dpPrev_input:
			last_tucker_rank += 1
		if add_uDotGradDpPrev_input:
			last_tucker_rank += 1
		if add_gradDpPrevMag_input:
			last_tucker_rank += 1
		if include_rAU_input:
			last_tucker_rank += 1
		if include_HbyA_input:
			last_tucker_rank += 3
		if include_divHbyA_input:
			last_tucker_rank += 1
		if include_dHbyA_input:
			last_tucker_rank += 3
		if include_dDivHbyA_input:
			last_tucker_rank += 1
		if add_rAUGradDpPrev_input:
			last_tucker_rank += 3
		if add_divRAUGradDpPrev_input:
			last_tucker_rank += 1
		if add_pressureEqResidualp_input:
			last_tucker_rank += 1
		if add_rAUGradpPrev_input:
			last_tucker_rank += 3
		if add_divRAUGradpPrev_input:
			last_tucker_rank += 1
		if add_divDDUStar_input:
			last_tucker_rank += 1
		if add_divDUStar_input:
			last_tucker_rank += 1
		if add_divUStar_input:
			last_tucker_rank += 1
		if add_distance_to_outlet_input:
			last_tucker_rank += 1
		if add_grad_sdf_input:
			last_tucker_rank += 3
		if add_UdotNwall_input:
			last_tucker_rank += 1
		if verbose:
			print(f'[load_tucker_and_NN] Auto-calculated last_tucker_rank: {last_tucker_rank}')

		# Compute and store the channel index of divUStar in the raw spatial block.
		# Used by cnn_multi_out_divu to build the hard mask inside the model.
		global div_u_ch_idx_g
		_div_u_ch = 0
		if add_U_input:
			_div_u_ch += 3
		if add_dUStar_input:
			_div_u_ch += 3
		if add_ddUStar_input:
			_div_u_ch += 3
		if add_ddUStarDiff_input:
			_div_u_ch += 3
		if add_dUCorrPrev_input:
			_div_u_ch += 3
		if add_ddUCorrPrev_input:
			_div_u_ch += 3
		# Note: sdf is in inputs_obst (last channel), NOT in inputs_u, so it is not counted here
		if add_pPrev_input:
			_div_u_ch += 1
		if add_dpPrev_input:
			_div_u_ch += 1
		if add_ddpPrev_input:
			_div_u_ch += 1
		if add_gradDpPrev_input:
			_div_u_ch += 3
		if add_laplacian_dpPrev_input:
			_div_u_ch += 1
		if add_uDotGradDpPrev_input:
			_div_u_ch += 1
		if add_gradDpPrevMag_input:
			_div_u_ch += 1
		if include_rAU_input:
			_div_u_ch += 1
		if include_HbyA_input:
			_div_u_ch += 3
		if include_divHbyA_input:
			_div_u_ch += 1
		if include_dHbyA_input:
			_div_u_ch += 3
		if include_dDivHbyA_input:
			_div_u_ch += 1
		if add_rAUGradDpPrev_input:
			_div_u_ch += 3
		if add_divRAUGradDpPrev_input:
			_div_u_ch += 1
		if add_pressureEqResidualp_input:
			_div_u_ch += 1
		if add_rAUGradpPrev_input:
			_div_u_ch += 3
		if add_divRAUGradpPrev_input:
			_div_u_ch += 1
		if add_divDDUStar_input:
			_div_u_ch += 1
		if add_divDUStar_input:
			_div_u_ch += 1
		div_u_ch_idx_g = _div_u_ch if add_divUStar_input else None
		if verbose:
			print(f'[load_tucker_and_NN] div_u_ch_idx_g (divUStar channel in raw block): {div_u_ch_idx_g}')

		if predict_ddUCorr:
			if use_feature_decomposition:
				raise ValueError('[load_tucker_and_NN] predict_ddUCorr=True is not supported with Tucker decomposition (use_feature_decomposition=True). Set use_feature_decomposition=False.')
			print(f'[load_tucker_and_NN] predict_ddUCorr=True: model will predict [ddp, ddUx, ddUy, ddUz] (4 output channels)')
	
		# Store Tucker rank info so init_func can build the MLP with correct sizes
		global spatial_tucker_ranks_g, last_tucker_rank_g
		spatial_tucker_ranks_g = spatial_tucker_ranks
		last_tucker_rank_g = last_tucker_rank

		if verbose:
			print(f'[load_tucker_and_NN] Configuration stored. Model will be created in init_func.')
			print(f'[load_tucker_and_NN] overlap_ratio: {overlap_ratio}')
			print(f'[load_tucker_and_NN] add_U_input: {add_U_input}, add_dUStar_input: {add_dUStar_input}, add_ddUStar_input: {add_ddUStar_input}, add_ddUStarDiff_input: {add_ddUStarDiff_input}, add_dUCorrPrev_input: {add_dUCorrPrev_input}, add_ddUCorrPrev_input: {add_ddUCorrPrev_input}, add_dpPrev_input: {add_dpPrev_input}, add_pPrev_input: {add_pPrev_input}, add_ddpPrev_input: {add_ddpPrev_input}, add_gradDpPrev_input: {add_gradDpPrev_input}, add_laplacian_dpPrev_input: {add_laplacian_dpPrev_input}, add_uDotGradDpPrev_input: {add_uDotGradDpPrev_input}, add_gradDpPrevMag_input: {add_gradDpPrevMag_input}, add_divDDUStar_input: {add_divDDUStar_input}, add_divDUStar_input: {add_divDUStar_input}, add_divUStar_input: {add_divUStar_input}, add_distance_to_outlet_input: {add_distance_to_outlet_input}, add_grad_sdf_input: {add_grad_sdf_input}, add_UdotNwall_input: {add_UdotNwall_input}, clip_UdotNwall_to_inflow: {clip_UdotNwall_to_inflow}, last_tucker_rank: {last_tucker_rank}, predict_ddUCorr: {predict_ddUCorr}')


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

		print('Running init function... This might take a while! ', flush=True)
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

		indices = np.zeros((X0.shape[0], 3), dtype=np.int64)  # zeros: non-domain points default to (0,0,0) instead of garbage
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

		# Compute static geometric grids for optional inputs
		global dist_to_outlet_grid, grad_sdf_grid, UdotNwall_grid
		if add_distance_to_outlet_input_g:
			# Distance to the outlet: use x_max face as outlet proxy (physical x = axis 0 of xyz0)
			_x_max = float(np.max(X0))
			_dist_flat = _x_max - X0  # distance from each grid point to the outlet (x_max face)
			dist_to_outlet_grid = np.zeros((grid_shape_z, grid_shape_y, grid_shape_x, 1), dtype=np.float64)
			dist_to_outlet_grid[indices_i, indices_j, indices_k, 0] = _dist_flat
			# zero outside domain (domain points have non-zero sdf)
			dist_to_outlet_grid *= (sdfunct != 0)
			print(f'[init_func] dist_to_outlet_grid computed: max={dist_to_outlet_grid.max():.4f}')
		else:
			dist_to_outlet_grid = None

		if add_grad_sdf_input_g or add_UdotNwall_input_g:
			_sdf_3d = sdfunct[:, :, :, 0]
			_grad_sdf_z_3d, _grad_sdf_y_3d, _grad_sdf_x_3d = np.gradient(_sdf_3d, dz, dy, dx)
			# Stack in physical [x, y, z] component order to match maxs layout
			grad_sdf_grid = np.stack([_grad_sdf_x_3d, _grad_sdf_y_3d, _grad_sdf_z_3d], axis=-1)  # (z, y, x, 3)
			print(f'[init_func] grad_sdf_grid computed: max_x={np.abs(_grad_sdf_x_3d).max():.4f}, max_y={np.abs(_grad_sdf_y_3d).max():.4f}, max_z={np.abs(_grad_sdf_z_3d).max():.4f}')
		else:
			grad_sdf_grid = None

		if add_UdotNwall_input_g:
			# UdotNwall_grid will be computed in py_func once U is available
			# For now, just declare it as a global that will be set during execution
			pass
		else:
			UdotNwall_grid = None

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
			_n_out_ch = 4 if predict_ddUCorr_g else 1
			print(f'[init_func] Creating {arch} model with block shape {_bs}, out_channels={_n_out_ch}')
			if arch == 'cnn':
				model = SimpleCNN3D(_bs, in_channels=last_tucker_rank_g, out_channels=_n_out_ch, dropout_rate=dropout_rate_g, regularization=regularization_g)
			elif arch == 'cnn_two_heads':
				model = SimpleCNN3D_two_heads(_bs, in_channels=last_tucker_rank_g,
								  return_heads=True,
								  dropout_rate=dropout_rate_g, regularization=regularization_g)
			elif arch == 'cnn_two_heads_smooth':
				model = SimpleCNN3D_two_heads_smooth(_bs, in_channels=last_tucker_rank_g,
								  return_heads=True,
								  dropout_rate=dropout_rate_g, regularization=regularization_g)
			elif arch == 'cnn_multi_out':
				model = SimpleCNN3D_multi_out(_bs, in_channels=last_tucker_rank_g, out_channels=_n_out_ch, dropout_rate=dropout_rate_g, regularization=regularization_g)
			elif arch == 'cnn_multi_out_divu':
				if div_u_ch_idx_g is None:
					raise ValueError('[init_func] cnn_multi_out_divu requires add_divUStar_input=True in python_module.py')
				# Extract per-channel normalization factors for divU so the model can
				# denormalize before binarizing the mask (physical zeros → physical 0).
				_divu_mean = float(mean_in.flat[div_u_ch_idx_g]) if mean_in is not None else 0.0
				_divu_std  = float(std_in.flat[div_u_ch_idx_g])  if std_in  is not None else 1.0
				model = SimpleCNN3D_multi_out_divU(_bs, in_channels=last_tucker_rank_g, out_channels=_n_out_ch, dropout_rate=dropout_rate_g, regularization=regularization_g, div_u_ch_idx=div_u_ch_idx_g, div_u_mean=_divu_mean, div_u_std=_divu_std)
			elif arch == 'cnn_shifter' or arch == 'simplecnn3d_ddp_shifter':
				model = SimpleCNN3D_ddp_shifter(_bs, in_channels=last_tucker_rank_g, dropout_rate=dropout_rate_g, regularization=regularization_g)
			elif arch == 'cnn_shifter_lightweight' or arch == 'simplecnn3d_ddp_shifter_lightweight':
				model = SimpleCNN3D_ddp_shifter_lightweight(_bs, in_channels=last_tucker_rank_g, dropout_rate=dropout_rate_g, regularization=regularization_g)
			elif arch == 'cnn_shifter_velocity' or arch == 'simplecnn3d_ddp_shifter_velocity':
				model = SimpleCNN3D_ddp_shifter_velocity(_bs, in_channels=last_tucker_rank_g, dropout_rate=dropout_rate_g, regularization=regularization_g)
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
				model = SimpleCNN3D(_bs, in_channels=last_tucker_rank_g, out_channels=_n_out_ch, dropout_rate=dropout_rate_g, regularization=regularization_g)
		model.load_weights(weights_fn_g)
		print(f'[init_func] Model loaded from {weights_fn_g}')

		# === LOAD ORACLE DATA (if enabled) ===
		if oracle_mode_g:
			import h5py
			global oracle_ddp_cache_g, oracle_ddu_cache_g
			oracle_h5_path = os.path.join(oracle_data_folder_g, 'data.h5')
			if os.path.exists(oracle_h5_path):
				try:
					with h5py.File(oracle_h5_path, 'r') as f_oracle:
						# Get all sample keys in sorted order
						sample_keys = sorted([key for key in f_oracle.keys() if key.startswith('sample_')])
						print(f'[oracle] Loading {len(sample_keys)} samples from {oracle_h5_path}')
						
						oracle_ddp_cache_g = []
						oracle_ddu_cache_g = []
						for sample_key in sample_keys:
							group = f_oracle[sample_key]
							# Find the pressure key - consistent with load_hdf5_samples
							ddp_key = 'ddp' if 'ddp' in group else 'pressure_increment'
							
							if ddp_key in group:
								# Load pressure as numpy array - already at OF mesh points
								ddpML_value = np.array(group[ddp_key][:], dtype=np.float64)
								oracle_ddp_cache_g.append(ddpML_value)
							else:
								oracle_ddp_cache_g.append(None)
							
							# Load velocity if available (for predict_ddUCorr mode)
							ddu_key = 'delta_delta_U_CFD' if 'delta_delta_U_CFD' in group else 'ddU'
							if ddu_key in group:
								ddu_value = np.array(group[ddu_key][:], dtype=np.float64)  # (n_cells, 3)
								oracle_ddu_cache_g.append(ddu_value)
							else:
								oracle_ddu_cache_g.append(None)
						
						print(f'[oracle] Loaded {len(oracle_ddp_cache_g)} oracle samples (pressure + velocity)')
				except Exception as e:
					print(f'[oracle] WARNING: Failed to load oracle HDF5 data: {e}')
					oracle_ddp_cache_g = []
					oracle_ddu_cache_g = []
			else:
				print(f'[oracle] WARNING: Oracle HDF5 file not found at {oracle_h5_path}')

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
	  Channels 3-5:   dU (velocity increment, if add_dUStar_input_g=True)
	  Channels 6-8:   ddU (second velocity increment, if add_ddUStar_input_g=True)
	  Channels 9-11:  dddU (velocity difference, if add_ddUStarDiff_input_g=True)
	  Channel 12:     p_prev (previous pressure, if add_pPrev_input_g=True)
	  Channel 13:     dpPrev (previous pressure increment, if add_dpPrev_input_g=True)
	  Channel 14:     ddpPrev (second pressure increment, if add_ddpPrev_input_g=True)
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
		# Array layout (15 cols): 0-2 U, 3-5 ddU, 6-8 ddU_prev, 9 ddpPrev, 10 dpPrev, 11-13 dU, 14 p_rgh_prev

		ch_idx = 0
		if add_U_input_g or add_UdotNwall_input_g:
			# U is always at channels 0-2 in the C++ array; needed for grid channel and/or UdotNwall
			U = array[..., 0:3]
			if add_U_input_g:
				ch_idx += 3
		else:
			U = None
		
		delta_delta_U = array[..., 3:6]
		delta_delta_U_prev = array[..., 6:9]
		if add_ddUStar_input_g:
			ch_idx += 3
		
		# dddU (optional)
		delta_delta_U_diff = delta_delta_U - delta_delta_U_prev
		if add_ddUStarDiff_input_g:
			ch_idx += 3
		
		# Channel 9: ddpPrev (SM input if add_ddpPrev_input, else not used)
		ddpPrev = array[..., 9:10]

		# Channel 10: dpPrev — SM input (if add_dpPrev_input)
		if add_dpPrev_input_g:
			dpPrev = array[..., 10:11]
			ch_idx += 1
		else:
			dpPrev = None

		if add_gradDpPrev_input_g:
			gradDpPrev = array[..., 22:25]
		else:
			gradDpPrev = None

		if add_laplacian_dpPrev_input_g:
			laplaceDpPrev = array[..., 25:26]
		else:
			laplaceDpPrev = None

		if add_uDotGradDpPrev_input_g:
			uDotGradDpPrev = array[..., 26:27]
		else:
			uDotGradDpPrev = None

		if add_gradDpPrevMag_input_g:
			gradDpPrevMag = array[..., 27:28]
		else:
			gradDpPrevMag = None

		# PISO pressure-equation inputs (new C++ raw columns 28-32)
		if include_rAU_input_g:
			rAU = array[..., 28:29]
		else:
			rAU = None

		if include_HbyA_input_g:
			HbyA = array[..., 29:32]
		else:
			HbyA = None

		if include_divHbyA_input_g:
			divHbyA = array[..., 32:33]
		else:
			divHbyA = None

		# Temporal-variation PISO inputs (new C++ raw columns 33-36)
		if include_dHbyA_input_g:
			dHbyA = array[..., 33:36]
		else:
			dHbyA = None

		if include_dDivHbyA_input_g:
			dDivHbyA = array[..., 36:37]
		else:
			dDivHbyA = None

		if add_rAUGradDpPrev_input_g:
			rAUGradDpPrev = array[..., 37:40]
		else:
			rAUGradDpPrev = None

		if add_divRAUGradDpPrev_input_g:
			divRAUGradDpPrev = array[..., 40:41]
		else:
			divRAUGradDpPrev = None

		if add_pressureEqResidualp_input_g:
			pressureEqResidualp = array[..., 41:42]
		else:
			pressureEqResidualp = None

		if add_rAUGradpPrev_input_g:
			rAUGradpPrev = array[..., 42:45]
		else:
			rAUGradpPrev = None

		if add_divRAUGradpPrev_input_g:
			divRAUGradpPrev = array[..., 45:46]
		else:
			divRAUGradpPrev = None

		# Channels 11-13: delta_U (first velocity increment)
		if add_dUStar_input_g:
			delta_U = array[..., 11:14]
			ch_idx += 3
		else:
			delta_U = None

		# Channel 14: p_rgh_prev (absolute previous pressure)
		if add_pPrev_input_g:
			p_rgh_prev = array[..., 14:15]
			ch_idx += 1
		else:
			p_rgh_prev = None

		# Channel 15: divUFirstPred — only this div field exists in the new C++ layout (replaces divDDUStar/divDUStar/divUStar)
		if add_divDDUStar_input_g:
			div_ddu = array[..., 15:16]  # NOTE: this position is now divUFirstPred; add_divDDUStar_input_g should always be False
			ch_idx += 1
		else:
			div_ddu = None
		
		if add_divDUStar_input_g:
			div_du = array[..., 15:16]  # NOTE: divDUStar no longer exists; add_divDUStar_input_g should always be False
			ch_idx += 1
		else:
			div_du = None

		if add_divUStar_input_g:
			div_u = array[..., 15:16]  # divUFirstPred at channel 15 in new layout
			ch_idx += 1
		else:
			div_u = None

		# Channels 16-18: dUCorrPrev (previous pressure-correction velocity increment)
		if add_dUCorrPrev_input_g:
			dUCorrPrev = array[..., 16:19]
			ch_idx += 3
		else:
			dUCorrPrev = None

		# Channels 19-21: ddUCorrPrev (previous second pressure-correction velocity increment)
		if add_ddUCorrPrev_input_g:
			ddUCorrPrev = array[..., 19:22]
			ch_idx += 3
		else:
			ddUCorrPrev = None

		delta_U_changed = np.abs(delta_delta_U).sum(axis=-1)/np.abs(delta_delta_U).max()
		delta_delta_U_changed = np.abs(delta_delta_U - delta_delta_U_prev).sum(axis=-1)
        
		#delta_delta_U_changed = np.abs(delta_delta_U - delta_delta_U_prev).sum(axis=-1)
		#if delta_delta_U_changed.max() > 0:
		#	delta_delta_U_changed /= delta_delta_U_changed.max()

		# Normalize by U_max_norm
		if add_U_input_g:
			U_adim = U / U_max_norm  # U normalized
		if add_UdotNwall_input_g and not add_U_input_g:
			# UdotNwall is built from normalized U in training; keep inference consistent.
			U_adim = U / U_max_norm
		if add_dUStar_input_g:
			delta_U_adim = delta_U / U_max_norm  # dU normalized
		if add_ddUStar_input_g:
			delta_delta_U_adim = delta_delta_U / U_max_norm  # ddU normalized
		if add_ddUStarDiff_input_g:
			delta_delta_U_diff_adim = delta_delta_U_diff / U_max_norm  # dddU normalized
		else:
			delta_delta_U_diff_adim = None
		
		# Normalize dpPrev (SM input) if present
		if add_dpPrev_input_g:
			dpPrev_adim = dpPrev / (U_max_norm ** 2.0)
		else:
			dpPrev_adim = None

		# Normalize p_rgh_prev if present
		if add_pPrev_input_g:
			p_rgh_prev_adim = p_rgh_prev / (U_max_norm ** 2.0)
		else:
			p_rgh_prev_adim = None

		# Normalize ddpPrev (SM input) if add_ddpPrev_input is enabled
		if add_ddpPrev_input_g:
			ddpPrev_adim = ddpPrev / (U_max_norm ** 2.0)
		else:
			ddpPrev_adim = None

		if add_gradDpPrev_input_g:
			gradDpPrev_adim = gradDpPrev / (U_max_norm ** 2.0)
		else:
			gradDpPrev_adim = None

		if add_laplacian_dpPrev_input_g:
			laplaceDpPrev_adim = laplaceDpPrev / (U_max_norm ** 2.0)
		else:
			laplaceDpPrev_adim = None

		if add_uDotGradDpPrev_input_g:
			uDotGradDpPrev_adim = uDotGradDpPrev / (U_max_norm ** 3.0)
		else:
			uDotGradDpPrev_adim = None

		if add_gradDpPrevMag_input_g:
			gradDpPrevMag_adim = gradDpPrevMag / (U_max_norm ** 2.0)
		else:
			gradDpPrevMag_adim = None

		# PISO pressure-equation inputs: rAU (no U scaling), HbyA (/U_max_norm), divHbyA (/U_max_norm)
		if include_rAU_input_g:
			rAU_adim = rAU
		else:
			rAU_adim = None

		if include_HbyA_input_g:
			HbyA_adim = HbyA / U_max_norm
		else:
			HbyA_adim = None

		if include_divHbyA_input_g:
			divHbyA_adim = divHbyA / U_max_norm
		else:
			divHbyA_adim = None

		# Temporal-variation PISO inputs: dHbyA (/U_max_norm), dDivHbyA (/U_max_norm)
		if include_dHbyA_input_g:
			dHbyA_adim = dHbyA / U_max_norm
		else:
			dHbyA_adim = None

		if include_dDivHbyA_input_g:
			dDivHbyA_adim = dDivHbyA / U_max_norm
		else:
			dDivHbyA_adim = None

		if add_rAUGradDpPrev_input_g:
			rAUGradDpPrev_adim = rAUGradDpPrev / U_max_norm
		else:
			rAUGradDpPrev_adim = None

		if add_divRAUGradDpPrev_input_g:
			divRAUGradDpPrev_adim = divRAUGradDpPrev / U_max_norm
		else:
			divRAUGradDpPrev_adim = None

		if add_pressureEqResidualp_input_g:
			pressureEqResidualp_adim = pressureEqResidualp / U_max_norm
		else:
			pressureEqResidualp_adim = None

		if add_rAUGradpPrev_input_g:
			rAUGradpPrev_adim = rAUGradpPrev / U_max_norm
		else:
			rAUGradpPrev_adim = None

		if add_divRAUGradpPrev_input_g:
			divRAUGradpPrev_adim = divRAUGradpPrev / U_max_norm
		else:
			divRAUGradpPrev_adim = None
		
		if add_divDDUStar_input_g:
			div_ddu_adim = div_ddu / U_max_norm
		else:
			div_ddu_adim = None

		if add_divDUStar_input_g:
			div_du_adim = div_du / U_max_norm
		else:
			div_du_adim = None
		
		if add_divUStar_input_g:
			div_u_adim = div_u / U_max_norm
		else:
			div_u_adim = None

		if add_dUCorrPrev_input_g:
			dUCorrPrev_adim = dUCorrPrev / U_max_norm
		else:
			dUCorrPrev_adim = None

		if add_ddUCorrPrev_input_g:
			ddUCorrPrev_adim = ddUCorrPrev / U_max_norm
		else:
			ddUCorrPrev_adim = None

		if verbose_g: 
			print(f"Data pre-processing: {time.time()-t0} s")

		t0 = time.time()


		# Interpolate all components based on flags
		if add_U_input_g:
			U_x_interp = interpolate_fill_njit(U_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
			U_y_interp = interpolate_fill_njit(U_adim[:, 1], vert_OFtoNP_array, weights_OFtoNP_array)
			U_z_interp = interpolate_fill_njit(U_adim[:, 2], vert_OFtoNP_array, weights_OFtoNP_array)
		# For UdotNwall: interpolate normalized U (U/U_max_norm), consistent with training.
		if add_UdotNwall_input_g:
			_U_x_raw_interp = interpolate_fill_njit(U_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
			_U_y_raw_interp = interpolate_fill_njit(U_adim[:, 1], vert_OFtoNP_array, weights_OFtoNP_array)
			_U_z_raw_interp = interpolate_fill_njit(U_adim[:, 2], vert_OFtoNP_array, weights_OFtoNP_array)

		if add_dUStar_input_g:
			dU_x_interp = interpolate_fill_njit(delta_U_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
			dU_y_interp = interpolate_fill_njit(delta_U_adim[:, 1], vert_OFtoNP_array, weights_OFtoNP_array)
			dU_z_interp = interpolate_fill_njit(delta_U_adim[:, 2], vert_OFtoNP_array, weights_OFtoNP_array)
		
		if add_ddUStar_input_g:
			ddU_x_interp = interpolate_fill_njit(delta_delta_U_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
			ddU_y_interp = interpolate_fill_njit(delta_delta_U_adim[:, 1], vert_OFtoNP_array, weights_OFtoNP_array)
			ddU_z_interp = interpolate_fill_njit(delta_delta_U_adim[:, 2], vert_OFtoNP_array, weights_OFtoNP_array)
		
		if add_ddUStarDiff_input_g:
			dddU_x_interp = interpolate_fill_njit(delta_delta_U_diff_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
			dddU_y_interp = interpolate_fill_njit(delta_delta_U_diff_adim[:, 1], vert_OFtoNP_array, weights_OFtoNP_array)
			dddU_z_interp = interpolate_fill_njit(delta_delta_U_diff_adim[:, 2], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			dddU_x_interp = dddU_y_interp = dddU_z_interp = None
		delta_U_changed_interp = interpolate_fill_njit(delta_U_changed, vert_OFtoNP_array, weights_OFtoNP_array)
		
		# Interpolate dpPrev (SM input) and ddpPrev
		if add_dpPrev_input_g:
			dpPrev_interp = interpolate_fill_njit(dpPrev_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			dpPrev_interp = None

		ddpPrev_interp_to_plot = interpolate_fill_njit(ddpPrev[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
		
		if add_ddpPrev_input_g:
			ddpPrev_interp = interpolate_fill_njit(ddpPrev_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			ddpPrev_interp = None

		if add_gradDpPrev_input_g:
			gradDpPrev_x_interp = interpolate_fill_njit(gradDpPrev_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
			gradDpPrev_y_interp = interpolate_fill_njit(gradDpPrev_adim[:, 1], vert_OFtoNP_array, weights_OFtoNP_array)
			gradDpPrev_z_interp = interpolate_fill_njit(gradDpPrev_adim[:, 2], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			gradDpPrev_x_interp = gradDpPrev_y_interp = gradDpPrev_z_interp = None

		if add_laplacian_dpPrev_input_g:
			laplaceDpPrev_interp = interpolate_fill_njit(laplaceDpPrev_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			laplaceDpPrev_interp = None

		if add_uDotGradDpPrev_input_g:
			uDotGradDpPrev_interp = interpolate_fill_njit(uDotGradDpPrev_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			uDotGradDpPrev_interp = None

		if add_gradDpPrevMag_input_g:
			gradDpPrevMag_interp = interpolate_fill_njit(gradDpPrevMag_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			gradDpPrevMag_interp = None

		if include_rAU_input_g:
			rAU_interp = interpolate_fill_njit(rAU_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			rAU_interp = None

		if include_HbyA_input_g:
			HbyA_x_interp = interpolate_fill_njit(HbyA_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
			HbyA_y_interp = interpolate_fill_njit(HbyA_adim[:, 1], vert_OFtoNP_array, weights_OFtoNP_array)
			HbyA_z_interp = interpolate_fill_njit(HbyA_adim[:, 2], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			HbyA_x_interp = HbyA_y_interp = HbyA_z_interp = None

		if include_divHbyA_input_g:
			divHbyA_interp = interpolate_fill_njit(divHbyA_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			divHbyA_interp = None

		if include_dHbyA_input_g:
			dHbyA_x_interp = interpolate_fill_njit(dHbyA_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
			dHbyA_y_interp = interpolate_fill_njit(dHbyA_adim[:, 1], vert_OFtoNP_array, weights_OFtoNP_array)
			dHbyA_z_interp = interpolate_fill_njit(dHbyA_adim[:, 2], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			dHbyA_x_interp = dHbyA_y_interp = dHbyA_z_interp = None

		if include_dDivHbyA_input_g:
			dDivHbyA_interp = interpolate_fill_njit(dDivHbyA_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			dDivHbyA_interp = None

		if add_rAUGradDpPrev_input_g:
			rAUGradDpPrev_x_interp = interpolate_fill_njit(rAUGradDpPrev_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
			rAUGradDpPrev_y_interp = interpolate_fill_njit(rAUGradDpPrev_adim[:, 1], vert_OFtoNP_array, weights_OFtoNP_array)
			rAUGradDpPrev_z_interp = interpolate_fill_njit(rAUGradDpPrev_adim[:, 2], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			rAUGradDpPrev_x_interp = rAUGradDpPrev_y_interp = rAUGradDpPrev_z_interp = None

		if add_divRAUGradDpPrev_input_g:
			divRAUGradDpPrev_interp = interpolate_fill_njit(divRAUGradDpPrev_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			divRAUGradDpPrev_interp = None

		if add_pressureEqResidualp_input_g:
			pressureEqResidualp_interp = interpolate_fill_njit(pressureEqResidualp_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			pressureEqResidualp_interp = None

		if add_rAUGradpPrev_input_g:
			rAUGradpPrev_x_interp = interpolate_fill_njit(rAUGradpPrev_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
			rAUGradpPrev_y_interp = interpolate_fill_njit(rAUGradpPrev_adim[:, 1], vert_OFtoNP_array, weights_OFtoNP_array)
			rAUGradpPrev_z_interp = interpolate_fill_njit(rAUGradpPrev_adim[:, 2], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			rAUGradpPrev_x_interp = rAUGradpPrev_y_interp = rAUGradpPrev_z_interp = None

		if add_divRAUGradpPrev_input_g:
			divRAUGradpPrev_interp = interpolate_fill_njit(divRAUGradpPrev_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			divRAUGradpPrev_interp = None

		if add_pPrev_input_g:
			p_rgh_prev_interp = interpolate_fill_njit(p_rgh_prev_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			p_rgh_prev_interp = None

		if add_divDDUStar_input_g:
			div_ddu_interp = interpolate_fill_njit(div_ddu_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			div_ddu_interp = None
		
		if add_divDUStar_input_g:
			div_du_interp = interpolate_fill_njit(div_du_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			div_du_interp = None

		if add_divUStar_input_g:
			div_u_interp = interpolate_fill_njit(div_u_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			div_u_interp = None

		if add_dUCorrPrev_input_g:
			dUCorrPrev_x_interp = interpolate_fill_njit(dUCorrPrev_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
			dUCorrPrev_y_interp = interpolate_fill_njit(dUCorrPrev_adim[:, 1], vert_OFtoNP_array, weights_OFtoNP_array)
			dUCorrPrev_z_interp = interpolate_fill_njit(dUCorrPrev_adim[:, 2], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			dUCorrPrev_x_interp = dUCorrPrev_y_interp = dUCorrPrev_z_interp = None

		if add_ddUCorrPrev_input_g:
			ddUCorrPrev_x_interp = interpolate_fill_njit(ddUCorrPrev_adim[:, 0], vert_OFtoNP_array, weights_OFtoNP_array)
			ddUCorrPrev_y_interp = interpolate_fill_njit(ddUCorrPrev_adim[:, 1], vert_OFtoNP_array, weights_OFtoNP_array)
			ddUCorrPrev_z_interp = interpolate_fill_njit(ddUCorrPrev_adim[:, 2], vert_OFtoNP_array, weights_OFtoNP_array)
		else:
			ddUCorrPrev_x_interp = ddUCorrPrev_y_interp = ddUCorrPrev_z_interp = None

		if verbose_g:
			print(f"1st interpolation took: {time.time()-t0} s")


		t0 = time.time()
		# Grid channels: [U(3 if add_U)] [dU(3 if add_dU)] [ddU(3 if add_ddu)] [dddU(3 if add_dddu)] [dUCorrPrev(3 if add_dUCorrPrev)] [ddUCorrPrev(3 if add_ddUCorrPrev)] [p_prev(1 if add_p_prev)] [dpPrev(1 if add_dpPrev)] [ddpPrev(1 if add_ddpPrev)] [dist_to_outlet(1 if add_distance_to_outlet)] [grad_sdf(3 if add_grad_sdf)] [U*nwall(1 if add_UdotNwall)] sdf(1)
		n_grid_ch = (3 if add_U_input_g else 0) + (3 if add_dUStar_input_g else 0) + (3 if add_ddUStar_input_g else 0) + (3 if add_ddUStarDiff_input_g else 0) + (3 if add_dUCorrPrev_input_g else 0) + (3 if add_ddUCorrPrev_input_g else 0) + (1 if add_pPrev_input_g else 0) + (1 if add_dpPrev_input_g else 0) + (1 if add_ddpPrev_input_g else 0) + (3 if add_gradDpPrev_input_g else 0) + (1 if add_laplacian_dpPrev_input_g else 0) + (1 if add_uDotGradDpPrev_input_g else 0) + (1 if add_gradDpPrevMag_input_g else 0) + (1 if include_rAU_input_g else 0) + (3 if include_HbyA_input_g else 0) + (1 if include_divHbyA_input_g else 0) + (3 if include_dHbyA_input_g else 0) + (1 if include_dDivHbyA_input_g else 0) + (3 if add_rAUGradDpPrev_input_g else 0) + (1 if add_divRAUGradDpPrev_input_g else 0) + (1 if add_pressureEqResidualp_input_g else 0) + (3 if add_rAUGradpPrev_input_g else 0) + (1 if add_divRAUGradpPrev_input_g else 0) + (1 if add_divDDUStar_input_g else 0) + (1 if add_divDUStar_input_g else 0) + (1 if add_divUStar_input_g else 0) + (1 if add_distance_to_outlet_input_g else 0) + (3 if add_grad_sdf_input_g else 0) + (1 if add_UdotNwall_input_g else 0) + 1
		grid = np.zeros((grid_shape_z, grid_shape_y, grid_shape_x, n_grid_ch), dtype=np.float64)
		delta_U_change_grid = np.zeros((grid_shape_z, grid_shape_y, grid_shape_x), dtype=np.float64)

		# Build the interpolated stack based on what's included
		ch_idx = 0
		interp_parts = []
		if add_U_input_g:
			interp_parts.append(np.column_stack([U_x_interp, U_y_interp, U_z_interp]))
			ch_idx += 3
		if add_dUStar_input_g:
			interp_parts.append(np.column_stack([dU_x_interp, dU_y_interp, dU_z_interp]))
			ch_idx += 3
		if add_ddUStar_input_g:
			interp_parts.append(np.column_stack([ddU_x_interp, ddU_y_interp, ddU_z_interp]))
			ch_idx += 3
		if add_ddUStarDiff_input_g:
			interp_parts.append(np.column_stack([dddU_x_interp, dddU_y_interp, dddU_z_interp]))
			ch_idx += 3
		if add_dUCorrPrev_input_g:
			interp_parts.append(np.column_stack([dUCorrPrev_x_interp, dUCorrPrev_y_interp, dUCorrPrev_z_interp]))
			ch_idx += 3
		if add_ddUCorrPrev_input_g:
			interp_parts.append(np.column_stack([ddUCorrPrev_x_interp, ddUCorrPrev_y_interp, ddUCorrPrev_z_interp]))
			ch_idx += 3
		if add_pPrev_input_g:
			interp_parts.append(p_rgh_prev_interp[:, np.newaxis])  # p_prev channel
			ch_idx += 1
		if add_dpPrev_input_g:
			interp_parts.append(dpPrev_interp[:, np.newaxis])  # dpPrev channel
			ch_idx += 1
		if add_ddpPrev_input_g:
			interp_parts.append(ddpPrev_interp[:, np.newaxis])  # ddpPrev channel
			ch_idx += 1
		if add_gradDpPrev_input_g:
			interp_parts.append(np.column_stack([gradDpPrev_x_interp, gradDpPrev_y_interp, gradDpPrev_z_interp]))
			ch_idx += 3
		if add_laplacian_dpPrev_input_g:
			interp_parts.append(laplaceDpPrev_interp[:, np.newaxis])
			ch_idx += 1
		if add_uDotGradDpPrev_input_g:
			interp_parts.append(uDotGradDpPrev_interp[:, np.newaxis])
			ch_idx += 1
		if add_gradDpPrevMag_input_g:
			interp_parts.append(gradDpPrevMag_interp[:, np.newaxis])
			ch_idx += 1
		if include_rAU_input_g:
			interp_parts.append(rAU_interp[:, np.newaxis])  # rAU channel
			ch_idx += 1
		if include_HbyA_input_g:
			interp_parts.append(np.column_stack([HbyA_x_interp, HbyA_y_interp, HbyA_z_interp]))  # HbyA channels
			ch_idx += 3
		if include_divHbyA_input_g:
			interp_parts.append(divHbyA_interp[:, np.newaxis])  # divHbyA channel
			ch_idx += 1
		if include_dHbyA_input_g:
			interp_parts.append(np.column_stack([dHbyA_x_interp, dHbyA_y_interp, dHbyA_z_interp]))  # dHbyA channels
			ch_idx += 3
		if include_dDivHbyA_input_g:
			interp_parts.append(dDivHbyA_interp[:, np.newaxis])  # dDivHbyA channel
			ch_idx += 1
		if add_rAUGradDpPrev_input_g:
			interp_parts.append(np.column_stack([rAUGradDpPrev_x_interp, rAUGradDpPrev_y_interp, rAUGradDpPrev_z_interp]))
			ch_idx += 3
		if add_divRAUGradDpPrev_input_g:
			interp_parts.append(divRAUGradDpPrev_interp[:, np.newaxis])
			ch_idx += 1
		if add_pressureEqResidualp_input_g:
			interp_parts.append(pressureEqResidualp_interp[:, np.newaxis])
			ch_idx += 1
		if add_rAUGradpPrev_input_g:
			interp_parts.append(np.column_stack([rAUGradpPrev_x_interp, rAUGradpPrev_y_interp, rAUGradpPrev_z_interp]))
			ch_idx += 3
		if add_divRAUGradpPrev_input_g:
			interp_parts.append(divRAUGradpPrev_interp[:, np.newaxis])
			ch_idx += 1
		if add_divDDUStar_input_g:
			interp_parts.append(div_ddu_interp[:, np.newaxis])  # div_ddu channel
			ch_idx += 1
		if add_divDUStar_input_g:
			interp_parts.append(div_du_interp[:, np.newaxis])  # div_du channel
			ch_idx += 1
		if add_divUStar_input_g:
			interp_parts.append(div_u_interp[:, np.newaxis])  # div_u channel
			ch_idx += 1
		interp_stack = np.column_stack(interp_parts)
		grid[indices_i, indices_j, indices_k, :ch_idx] = interp_stack
		if add_distance_to_outlet_input_g:
			grid[:, :, :, ch_idx] = dist_to_outlet_grid[:, :, :, 0]
			ch_idx += 1
		if add_grad_sdf_input_g:
			grid[:, :, :, ch_idx:ch_idx+3] = grad_sdf_grid  # (z, y, x, 3) channels in [x, y, z] order
			ch_idx += 3
		if add_UdotNwall_input_g:
			# Compute wall normal from grad(SDF): nwall = grad(SDF) / |grad(SDF)| + eps
			_grad_sdf_x = grad_sdf_grid[..., 0]
			_grad_sdf_y = grad_sdf_grid[..., 1]
			_grad_sdf_z = grad_sdf_grid[..., 2]
			_grad_sdf_mag = np.sqrt(_grad_sdf_x**2 + _grad_sdf_y**2 + _grad_sdf_z**2) + 1e-12
			_nwall_x = _grad_sdf_x / _grad_sdf_mag
			_nwall_y = _grad_sdf_y / _grad_sdf_mag
			_nwall_z = _grad_sdf_z / _grad_sdf_mag
			
			# Compute U dot nwall at mesh points using raw (unscaled) U, consistent with training
			U_dot_nwall_mesh = _U_x_raw_interp * _nwall_x[indices_i, indices_j, indices_k] + \
			                   _U_y_raw_interp * _nwall_y[indices_i, indices_j, indices_k] + \
			                   _U_z_raw_interp * _nwall_z[indices_i, indices_j, indices_k]
			
			# Fill grid at mesh points
			grid[indices_i, indices_j, indices_k, ch_idx] = U_dot_nwall_mesh
			
			if clip_UdotNwall_to_inflow_g:
				# max(-Un*, 0) — only keep positive inflow (negative U dot n: incoming flow)
				grid[:, :, :, ch_idx] = np.maximum(-grid[:, :, :, ch_idx], 0)
			
			ch_idx += 1
		grid[:, :, :, ch_idx] = sdfunct[:, :, :, 0]  # sdf is at position ch_idx

		delta_U_change_grid[indices_i, indices_j, indices_k] = delta_U_changed_interp

		# Build illustration grid for ddpPrev (channel 9)
		ddpPrev_grid = np.zeros((grid_shape_z, grid_shape_y, grid_shape_x), dtype=np.float64)
		ddpPrev_grid[indices_i, indices_j, indices_k] = ddpPrev_interp_to_plot

		# Normalization factors order: [U(3 if add_U)] [dU(3 if add_dU)] [ddU(3 if add_ddu)] dddU(3) [p_prev if add_p_prev] [dpPrev if add_dpPrev] [ddpPrev if add_ddpPrev] [div_ddu if add_div_ddu] [div_du if add_div_du] [div_u if add_div_u] sdf(1)
		norm_parts = []
		if add_U_input_g:
			norm_parts.extend([max_abs_U_x, max_abs_U_y, max_abs_U_z])
		if add_dUStar_input_g:
			norm_parts.extend([max_abs_dU_x, max_abs_dU_y, max_abs_dU_z])
		if add_ddUStar_input_g:
			norm_parts.extend([max_abs_ddU_x, max_abs_ddU_y, max_abs_ddU_z])
		if add_ddUStarDiff_input_g:
			norm_parts.extend([max_abs_dddU_x, max_abs_dddU_y, max_abs_dddU_z])  # dddU
		if add_dUCorrPrev_input_g:
			norm_parts.extend([max_abs_dUCorrPrev_x, max_abs_dUCorrPrev_y, max_abs_dUCorrPrev_z])  # dUCorrPrev
		if add_ddUCorrPrev_input_g:
			norm_parts.extend([max_abs_ddUCorrPrev_x, max_abs_ddUCorrPrev_y, max_abs_ddUCorrPrev_z])  # ddUCorrPrev
		if add_pPrev_input_g:
			norm_parts.append(max_abs_p_prev)  # p_prev
		if add_dpPrev_input_g:
			norm_parts.append(max_abs_dpPrev)  # dpPrev
		if add_ddpPrev_input_g:
			norm_parts.append(max_abs_ddpPrev)  # ddpPrev
		if add_gradDpPrev_input_g:
			norm_parts.extend([max_abs_gradDpPrev_x, max_abs_gradDpPrev_y, max_abs_gradDpPrev_z])  # gradDpPrev
		if add_laplacian_dpPrev_input_g:
			norm_parts.append(max_abs_laplacian_dpPrev)
		if add_uDotGradDpPrev_input_g:
			norm_parts.append(max_abs_uDotGradDpPrev)
		if add_gradDpPrevMag_input_g:
			norm_parts.append(max_abs_gradDpPrevMag)
		if include_rAU_input_g:
			norm_parts.append(max_abs_rAU)  # rAU
		if include_HbyA_input_g:
			norm_parts.extend([max_abs_HbyA_x, max_abs_HbyA_y, max_abs_HbyA_z])  # HbyA
		if include_divHbyA_input_g:
			norm_parts.append(max_abs_divHbyA)  # divHbyA
		if include_dHbyA_input_g:
			norm_parts.extend([max_abs_dHbyA_x, max_abs_dHbyA_y, max_abs_dHbyA_z])  # dHbyA
		if include_dDivHbyA_input_g:
			norm_parts.append(max_abs_dDivHbyA)  # dDivHbyA
		if add_rAUGradDpPrev_input_g:
			norm_parts.extend([max_abs_rAUGradDpPrev_x, max_abs_rAUGradDpPrev_y, max_abs_rAUGradDpPrev_z])
		if add_divRAUGradDpPrev_input_g:
			norm_parts.append(max_abs_divRAUGradDpPrev)
		if add_pressureEqResidualp_input_g:
			norm_parts.append(max_abs_pressureEqResidualp)
		if add_rAUGradpPrev_input_g:
			norm_parts.extend([max_abs_rAUGradpPrev_x, max_abs_rAUGradpPrev_y, max_abs_rAUGradpPrev_z])
		if add_divRAUGradpPrev_input_g:
			norm_parts.append(max_abs_divRAUGradpPrev)
		if add_divDDUStar_input_g:
			norm_parts.append(max_abs_div_ddu)  # div(delta_delta_U)
		if add_divDUStar_input_g:
			norm_parts.append(max_abs_div_du)  # div(delta_U)
		if add_divUStar_input_g:
			norm_parts.append(max_abs_div_u)  # div(U)
		if add_distance_to_outlet_input_g:
			norm_parts.append(max_abs_dist_to_outlet)  # distance to outlet
		if add_grad_sdf_input_g:
			norm_parts.extend([max_abs_grad_sdf_x, max_abs_grad_sdf_y, max_abs_grad_sdf_z])  # grad(sdf) [x, y, z]
		if add_UdotNwall_input_g:
			norm_parts.append(max_abs_UdotNwall)  # U dot nwall
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
						_vb = (3 if add_U_input_g else 0) + (3 if add_dUStar_input_g else 0) + (3 if add_ddUStar_input_g else 0) + (3 if add_ddUStarDiff_input_g else 0) + (3 if add_dUCorrPrev_input_g else 0) + (3 if add_ddUCorrPrev_input_g else 0)
						_dm = block[..., _sdf_ch] != 0
						if enforce_zero_mean_pressure_g and _dm.any():
							if add_pPrev_input_g:
								block[..., _vb][_dm] -= np.mean(block[..., _vb][_dm])
							_c = _vb + int(add_pPrev_input_g)
							if add_dpPrev_input_g:
								block[..., _c][_dm] -= np.mean(block[..., _c][_dm])
							if add_ddpPrev_input_g:
								_c2 = _c + int(add_dpPrev_input_g)
								block[..., _c2][_dm] -= np.mean(block[..., _c2][_dm])

					else:
						# Prepare block and pad if needed (as before)
						block = np.zeros((block_size_z, block_size_y, block_size_x, n_grid_ch), dtype=np.float64)
						block_z = z_f_clip - z_0_clip
						block_y = y_f_clip - y_0_clip
						block_x = x_f_clip - x_0_clip
						block[:block_z, :block_y, :block_x, :] = grid[z_0_clip:z_f_clip, y_0_clip:y_f_clip, x_0_clip:x_f_clip, :n_grid_ch]

						# Remove per-block domain mean from pressure input channels (sdf is last: n_grid_ch-1)
						_sdf_ch = n_grid_ch - 1
						_vb = (3 if add_U_input_g else 0) + (3 if add_dUStar_input_g else 0) + (3 if add_ddUStar_input_g else 0) + (3 if add_ddUStarDiff_input_g else 0) + (3 if add_dUCorrPrev_input_g else 0) + (3 if add_ddUCorrPrev_input_g else 0)
						_dm = block[..., _sdf_ch] != 0
						if enforce_zero_mean_pressure_g and _dm.any():
							if add_pPrev_input_g:
								block[..., _vb][_dm] -= np.mean(block[..., _vb][_dm])
							_c = _vb + int(add_pPrev_input_g)  # always defined so add_ddpPrev check below is safe
							if add_dpPrev_input_g:
								block[..., _c][_dm] -= np.mean(block[..., _c][_dm])
							if add_ddpPrev_input_g:
								_c2 = _c + int(add_dpPrev_input_g)
								block[..., _c2][_dm] -= np.mean(block[..., _c2][_dm])

					x_list[b] = block
					indices_list[b] = [i, j, n_x - 1 - k]
					b += 1

		x_array = x_list  # already an array

		# ------------------------------------------------------------
		# Optional compression with per-call threshold inside fluid domain
		# ------------------------------------------------------------
		clip_ddp_and_gradDpPrev = True
		clip_percentile = compression_clip_percentile_g

		fluid_mask_3d = sdfunct[:, :, :, 0] != 0  # True inside flow domain

		def compress_tail_asinh(arr3d, fluid_mask, q=95.0):
			vals = np.abs(arr3d[fluid_mask])
			if vals.size == 0:
				return arr3d, np.nan

			thr = np.percentile(vals, q)
			out = arr3d.copy()
			x = out[fluid_mask]
			ax = np.abs(x)
			mask = ax > thr
			out_vals = x.copy()
			excess = ax[mask] - thr
			compressed = thr + thr * np.arcsinh(excess / (thr + 1e-30))
			out_vals[mask] = np.sign(x[mask]) * compressed
			out[fluid_mask] = out_vals
			return out, thr

		if clip_ddp_and_gradDpPrev:
			# Compute ddpPrev channel index
			_vb = (3 if add_U_input_g else 0) + (3 if add_dUStar_input_g else 0) + (3 if add_ddUStar_input_g else 0) + (3 if add_ddUStarDiff_input_g else 0) + (3 if add_dUCorrPrev_input_g else 0) + (3 if add_ddUCorrPrev_input_g else 0)
			if add_ddpPrev_input_g:
				_c2 = _vb + int(add_pPrev_input_g) + int(add_dpPrev_input_g)  # ddpPrev channel index
				x_array[0, ..., _c2], _thr_ddpPrev = compress_tail_asinh(
					x_array[0, ..., _c2],
					fluid_mask_3d,
					q=clip_percentile,
				)


		# DEBUG: Verify channel count
		expected_channels = (
			(3 if add_U_input_g else 0)
			+ (3 if add_dUStar_input_g else 0)
			+ (3 if add_ddUStar_input_g else 0)
			+ (3 if add_ddUStarDiff_input_g else 0)
			+ (3 if add_dUCorrPrev_input_g else 0)
			+ (3 if add_ddUCorrPrev_input_g else 0)
			+ (1 if add_pPrev_input_g else 0)
			+ (1 if add_dpPrev_input_g else 0)
			+ (1 if add_ddpPrev_input_g else 0)
			+ 3 * int(add_gradDpPrev_input_g)
			+ int(add_laplacian_dpPrev_input_g)
			+ int(add_uDotGradDpPrev_input_g)
			+ int(add_gradDpPrevMag_input_g)
			+ int(include_rAU_input_g)
			+ 3 * int(include_HbyA_input_g)
			+ int(include_divHbyA_input_g)
			+ 3 * int(include_dHbyA_input_g)
			+ int(include_dDivHbyA_input_g)
			+ 3 * int(add_rAUGradDpPrev_input_g)
			+ int(add_divRAUGradDpPrev_input_g)
			+ int(add_pressureEqResidualp_input_g)
			+ 3 * int(add_rAUGradpPrev_input_g)
			+ int(add_divRAUGradpPrev_input_g)
			+ int(add_divDDUStar_input_g)
			+ int(add_divDUStar_input_g)
			+ int(add_divUStar_input_g)
			+ int(add_distance_to_outlet_input_g)
			+ 3 * int(add_grad_sdf_input_g)
			+ int(add_UdotNwall_input_g)
			+ 1  # sdf channel (LAST)
		)
		actual_channels = x_array.shape[-1]
		if verbose_g:
			print(f"[py_func DEBUG] x_array shape: {x_array.shape}")
			print(f"[py_func DEBUG] Expected channels: {expected_channels}, Actual channels: {actual_channels}")
			print(
				f"[py_func DEBUG] Input flags - "
				f"add_U: {add_U_input_g}, add_dU: {add_dUStar_input_g}, add_ddu: {add_ddUStar_input_g}, add_dddu: {add_ddUStarDiff_input_g}, "
				f"add_dUCorrPrev: {add_dUCorrPrev_input_g}, add_ddUCorrPrev: {add_ddUCorrPrev_input_g}, "
				f"add_p_prev: {add_pPrev_input_g}, add_dpPrev: {add_dpPrev_input_g}, add_ddpPrev: {add_ddpPrev_input_g}, "
				f"add_gradDpPrev: {add_gradDpPrev_input_g}, add_laplacian_dpPrev: {add_laplacian_dpPrev_input_g}, "
				f"add_uDotGradDpPrev: {add_uDotGradDpPrev_input_g}, add_gradDpPrevMag: {add_gradDpPrevMag_input_g}, "
				f"include_rAU: {include_rAU_input_g}, include_HbyA: {include_HbyA_input_g}, include_divHbyA: {include_divHbyA_input_g}, "
				f"include_dHbyA: {include_dHbyA_input_g}, include_dDivHbyA: {include_dDivHbyA_input_g}, "
				f"add_rAUGradDpPrev: {add_rAUGradDpPrev_input_g}, add_divRAUGradDpPrev: {add_divRAUGradDpPrev_input_g}, "
				f"add_pressureEqResidualp: {add_pressureEqResidualp_input_g}, add_rAUGradpPrev: {add_rAUGradpPrev_input_g}, "
				f"add_divRAUGradpPrev: {add_divRAUGradpPrev_input_g}, add_divDDUStar: {add_divDDUStar_input_g}, "
				f"add_divDUStar: {add_divDUStar_input_g}, add_divUStar: {add_divUStar_input_g}, "
				f"add_distance_to_outlet: {add_distance_to_outlet_input_g}, add_grad_sdf: {add_grad_sdf_input_g}, add_UdotNwall: {add_UdotNwall_input_g}"
			)
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
			x_input = (input_transformed - mean_in) / (std_in + 1e-8)
		else:
			# No Tucker: pass raw normalized blocks directly to the 3D CNN
			x_input = (x_array - mean_in) / (std_in + 1e-8)

		if verbose_g:
			print(f"{'Tucker transformation' if use_feature_decomposition_g else 'Input standardization'} : {time.time()-t0} s")


		t0 = time.time()
		arch = effective_model_arch_g.lower()
		if arch in ['cnn_two_heads', 'cnn_two_heads_smooth']:
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
		elif arch in ['cnn_shifter', 'cnn_shifter_lightweight', 'simplecnn3d_ddp_shifter', 'simplecnn3d_ddp_shifter_lightweight', 'cnn_shifter_velocity', 'simplecnn3d_ddp_shifter_velocity']:
			latent = np.array(model(x_input, training=False))
			if arch in ['cnn_shifter_velocity', 'simplecnn3d_ddp_shifter_velocity']:
				if not add_U_input_g:
					raise ValueError('[py_func] cnn_shifter_velocity requires add_U_input=True.')
				if gradDpPrev_input_ch_idxs_g is None:
					raise ValueError('[py_func] cnn_shifter_velocity requires add_gradDpPrev_input=True.')
				if U_input_ch_idxs_g is None:
					raise ValueError('[py_func] Failed to resolve U input channel indices.')
				ux_idx, uy_idx, uz_idx = U_input_ch_idxs_g
				gx_idx, gy_idx, gz_idx = gradDpPrev_input_ch_idxs_g
				mean_in_flat = np.ravel(mean_in)
				std_in_flat = np.ravel(std_in)
				mean_out_ddp = float(np.ravel(mean_out)[0])
				std_out_ddp = float(np.ravel(std_out)[0])
				u_x_common = (x_input[..., ux_idx:ux_idx + 1] * std_in_flat[ux_idx] + mean_in_flat[ux_idx]) * max_abs_U_x
				u_y_common = (x_input[..., uy_idx:uy_idx + 1] * std_in_flat[uy_idx] + mean_in_flat[uy_idx]) * max_abs_U_y
				u_z_common = (x_input[..., uz_idx:uz_idx + 1] * std_in_flat[uz_idx] + mean_in_flat[uz_idx]) * max_abs_U_z
				grad_x_common = (x_input[..., gx_idx:gx_idx + 1] * std_in_flat[gx_idx] + mean_in_flat[gx_idx]) * max_abs_gradDpPrev_x
				grad_y_common = (x_input[..., gy_idx:gy_idx + 1] * std_in_flat[gy_idx] + mean_in_flat[gy_idx]) * max_abs_gradDpPrev_y
				grad_z_common = (x_input[..., gz_idx:gz_idx + 1] * std_in_flat[gz_idx] + mean_in_flat[gz_idx]) * max_abs_gradDpPrev_z
				src_common = latent[..., 3:4] * std_out_ddp * max_abs_ddp
				ddp_common = (
					- latent[..., 0:1] * u_x_common * grad_x_common
					- latent[..., 1:2] * u_y_common * grad_y_common
					- latent[..., 2:3] * u_z_common * grad_z_common
					+ src_common
				)
				res_concat = (ddp_common / max_abs_ddp - mean_out_ddp) / std_out_ddp

				if verbose_g:
					_shift_x = -latent[..., 0:1] * u_x_common * grad_x_common
					_shift_y = -latent[..., 1:2] * u_y_common * grad_y_common
					_shift_z = -latent[..., 2:3] * u_z_common * grad_z_common
					print(
						"[shifter recon] "
						"|a_x|={:.3e} |a_y|={:.3e} |a_z|={:.3e} |src|={:.3e} || "
						"|u_x|={:.3e} |u_y|={:.3e} |u_z|={:.3e} || "
						"|grad_x|={:.3e} |grad_y|={:.3e} |grad_z|={:.3e}".format(
							float(np.mean(np.abs(latent[..., 0:1]))),
							float(np.mean(np.abs(latent[..., 1:2]))),
							float(np.mean(np.abs(latent[..., 2:3]))),
							float(np.mean(np.abs(latent[..., 3:4]))),
							float(np.mean(np.abs(u_x_common))),
							float(np.mean(np.abs(u_y_common))),
							float(np.mean(np.abs(u_z_common))),
							float(np.mean(np.abs(grad_x_common))),
							float(np.mean(np.abs(grad_y_common))),
							float(np.mean(np.abs(grad_z_common))),
						)
					)
					print(
						"[shifter recon] "
						"|shift_x|={:.3e} |shift_y|={:.3e} |shift_z|={:.3e} |src_common|={:.3e} || "
						"|ddp_common|={:.3e}  (max_abs_ddp={:.3e}, U_max_norm={:.3e})".format(
							float(np.mean(np.abs(_shift_x))),
							float(np.mean(np.abs(_shift_y))),
							float(np.mean(np.abs(_shift_z))),
							float(np.mean(np.abs(src_common))),
							float(np.mean(np.abs(ddp_common))),
							float(max_abs_ddp),
							float(U_max_norm),
						)
					)
			else:
				if gradDpPrev_input_ch_idxs_g is None:
					raise ValueError('[py_func] cnn_shifter requires add_gradDpPrev_input=True.')
				gx_idx, gy_idx, gz_idx = gradDpPrev_input_ch_idxs_g
				mean_in_flat = np.ravel(mean_in)
				std_in_flat = np.ravel(std_in)
				mean_out_ddp = float(np.ravel(mean_out)[0])
				std_out_ddp = float(np.ravel(std_out)[0])
				# Denormalize gradDpPrev to common/physical space, matching ShifterLoss(vector) in train.py:
				#   grad_common = (grad_norm * std_in + mean_in) * max_abs_grad
				grad_x_common = (x_input[..., gx_idx:gx_idx + 1] * std_in_flat[gx_idx] + mean_in_flat[gx_idx]) * max_abs_gradDpPrev_x
				grad_y_common = (x_input[..., gy_idx:gy_idx + 1] * std_in_flat[gy_idx] + mean_in_flat[gy_idx]) * max_abs_gradDpPrev_y
				grad_z_common = (x_input[..., gz_idx:gz_idx + 1] * std_in_flat[gz_idx] + mean_in_flat[gz_idx]) * max_abs_gradDpPrev_z
				# Source term in common space, matching ShifterLoss: src_common = s * std_out_ddp * max_abs_ddp
				src_common = latent[..., 3:4] * std_out_ddp * max_abs_ddp
				# Physics reconstruction in common space: ddp = -ux*grad_x - uy*grad_y - uz*grad_z + src
				ddp_common = (
					- latent[..., 0:1] * grad_x_common
					- latent[..., 1:2] * grad_y_common
					- latent[..., 2:3] * grad_z_common
					+ src_common
				)
				# Convert back to normalized ddp space so the shared downstream
				# (res_concat*std_out+mean_out) and (*max_abs_ddp*U_max_norm**2) pipeline is correct.
				res_concat = (ddp_common / max_abs_ddp - mean_out_ddp) / std_out_ddp

				if verbose_g:
					_shift_x = -latent[..., 0:1] * grad_x_common
					_shift_y = -latent[..., 1:2] * grad_y_common
					_shift_z = -latent[..., 2:3] * grad_z_common
					print(
						"[shifter recon vector] "
						"|ux|={:.3e} |uy|={:.3e} |uz|={:.3e} |src|={:.3e} || "
						"|grad_x|={:.3e} |grad_y|={:.3e} |grad_z|={:.3e}".format(
							float(np.mean(np.abs(latent[..., 0:1]))),
							float(np.mean(np.abs(latent[..., 1:2]))),
							float(np.mean(np.abs(latent[..., 2:3]))),
							float(np.mean(np.abs(latent[..., 3:4]))),
							float(np.mean(np.abs(grad_x_common))),
							float(np.mean(np.abs(grad_y_common))),
							float(np.mean(np.abs(grad_z_common))),
						)
					)
					print(
						"[shifter recon vector] "
						"|shift_x|={:.3e} |shift_y|={:.3e} |shift_z|={:.3e} |src_common|={:.3e} || "
						"|ddp_common|={:.3e}  (max_abs_ddp={:.3e}, U_max_norm={:.3e})".format(
							float(np.mean(np.abs(_shift_x))),
							float(np.mean(np.abs(_shift_y))),
							float(np.mean(np.abs(_shift_z))),
							float(np.mean(np.abs(src_common))),
							float(np.mean(np.abs(ddp_common))),
							float(max_abs_ddp),
							float(U_max_norm),
						)
					)
			res_concat = res_concat[..., 0]
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
		if predict_ddUCorr_g:
			# Multi-channel output: (N, bsz, bsy, bsx, 4)
			# Channel 0: ddp,  Channels 1-3: ddUx, ddUy, ddUz
			res_concat[..., 0] *= max_abs_ddp * pow(U_max_norm, 2.0)
			res_concat[..., 1] *= max_abs_delta_delta_U_x * U_max_norm
			res_concat[..., 2] *= max_abs_delta_delta_U_y * U_max_norm
			res_concat[..., 3] *= max_abs_delta_delta_U_z * U_max_norm
		else:
			res_concat *= max_abs_ddp * pow(U_max_norm, 2.0)
		if p_smooth_raw is not None:
			p_smooth_raw *= max_abs_ddp * pow(U_max_norm, 2.0)
			p_local_raw  *= max_abs_ddp * pow(U_max_norm, 2.0)

		number_of_nans = np.isnan(res_concat).sum()
		if verbose_g:
			print(f"Number of NaNs in res_concat before assembling: {number_of_nans}/{res_concat.size}")
		assert number_of_nans == 0, "NaN values found in res_concat before assembling."

		if verbose_g:
			print(f"Processing before assembly : {time.time()-t0} s", flush=True)

		# === ORACLE MODE: Short-circuit if oracle data available ===
		global _py_func_call_count
		if oracle_mode_g and len(oracle_ddp_cache_g) > 0 and rank == 0:
			if _py_func_call_count < len(oracle_ddp_cache_g):
				oracle_ddp = oracle_ddp_cache_g[_py_func_call_count]
				if oracle_ddp is not None:
					# Oracle data available - return it directly (already at mesh points)
					print(f'[oracle] Call {_py_func_call_count}: Using oracle ddp and optional delta_delta_U', flush=True)
					_py_func_call_count += 1
					# Split by rank and return
					if predict_ddUCorr_g:
						# Load velocity if available, otherwise use zeros
						oracle_ddu = oracle_ddu_cache_g[_py_func_call_count - 1] if _py_func_call_count - 1 < len(oracle_ddu_cache_g) else None
						if oracle_ddu is not None:
							# Use actual velocity from oracle
							oracle_full = np.column_stack([
								oracle_ddp,
								oracle_ddu[:, 0],  # ddUx
								oracle_ddu[:, 1],  # ddUy
								oracle_ddu[:, 2]   # ddUz
							])
						else:
							# No velocity data available, use zeros
							oracle_full = np.column_stack([
								oracle_ddp,
								np.zeros_like(oracle_ddp),
								np.zeros_like(oracle_ddp),
								np.zeros_like(oracle_ddp)
							])
						output_rankwise = np.split(oracle_full, np.cumsum(len_rankwise)[:-1])
					else:

						if oracle_model_with_interp_g:
							# To inpect the interpolation influence on results
							# 1. interpolate to grid -> 2.reinterpolate to mesh
							oracle_ddp_grid = interpolate_fill_njit(oracle_ddp, vert_OFtoNP_array, weights_OFtoNP_array)
							oracle_ddp = interpolate_fill_njit(oracle_ddp_grid, vert_NPtoOF_array, weights_NPtoOF_array)
						
						output_rankwise = np.split(oracle_ddp, np.cumsum(len_rankwise)[:-1])

					result = comm.scatter(output_rankwise, root=0)
					result = np.asarray(result, dtype=np.float64)
					print(output_weight_factor_g)
					return output_weight_factor_g * result

		t0 = time.time()

		Ref_BC = 0
		global _inspect_call_count
		if inspect_results_g:
			_inspect_call_count += 1

		_assemble_common = dict(
			indices_list=indices_list,
			n_x=n_x, n_y=n_y, n_z=n_z,
			overlap=overlap,
			shape=block_size,
			Ref_BC=Ref_BC,
			x_array=x_array,
			apply_filter=apply_filter_g,
			shape_x=grid_shape_x, shape_y=grid_shape_y, shape_z=grid_shape_z,
			delta_U_change_grid=delta_U_change_grid,
			ddpPrev_grid=ddpPrev_grid,
			apply_deltaU_change_wgt=False,
			filter_tuple=filter_tuple_g,
			filter_tuple_deltaU=(5, 5, 5),
			plot_results=inspect_results_g,
			plot_call_count=_inspect_call_count,
			plot_output_dir=inspect_output_dir_g,
		)

		if predict_ddUCorr_g:
			# Assemble each output channel separately: (N, bsz, bsy, bsx, 4) -> 4x (grid_z, grid_y, grid_x)
			# Each channel gets its own subdirectory so plots never overwrite each other.
			_ch_labels = ['ddp', 'ddUx', 'ddUy', 'ddUz']
			change_in_deltap_ch = []
			for _ch in range(4):
				_res_ch = res_concat[..., _ch]
				if _res_ch.dtype != np.float64:
					_res_ch = _res_ch.astype(np.float64, copy=False)
				_ch_common = dict(_assemble_common)
				_ch_common['plot_output_dir'] = os.path.join(inspect_output_dir_g, _ch_labels[_ch])
				_assembled = assemble_prediction(_res_ch, **_ch_common)
				change_in_deltap_ch.append(_assembled)
			# Stack into (grid_z, grid_y, grid_x, 4)
			change_in_deltap = np.stack(change_in_deltap_ch, axis=-1)
		else:
			# SPETIAL CASE: block representing the full simulation
			change_in_deltap = assemble_prediction(
					res_concat,
					**_assemble_common,
				)

		if verbose_g:
			print(f"Assembly algorithm took: {time.time()-t0} s", flush=True)

		# Inspect-mode: compare predicted ddP vs previous-step ddP on z-slices,
		# mirroring the train.py ddP comparison (both physical and common space).
		if inspect_results_g and not predict_ddUCorr_g:
			try:
				_plot_ddp_pred_vs_prev_z_slices(
					change_in_deltap,
					ddpPrev_grid,
					_inspect_call_count,
					inspect_output_dir_g,
					max_abs_ddp=max_abs_ddp,
					U_max_norm=U_max_norm,
				)
			except Exception as _e:
				print(f"[ddp_pred_vs_prev] WARNING: plotting failed: {_e}")

		# # Assemble and plot p_smooth / p_local grids (two-head model, inspect mode only)
		# if inspect_results_g and p_smooth_raw is not None:
		# 	_assemble_kwargs = dict(
		# 		indices_list=indices_list,
		# 		n_x=n_x, n_y=n_y, n_z=n_z,
		# 		overlap=overlap,
		# 		shape=block_size,
		# 		Ref_BC=Ref_BC,
		# 		x_array=x_array,
		# 		apply_filter=apply_filter_g,
		# 		shape_x=grid_shape_x, shape_y=grid_shape_y, shape_z=grid_shape_z,
		# 		delta_U_change_grid=delta_U_change_grid,
		# 		ddpPrev_grid=ddpPrev_grid,
		# 		apply_deltaU_change_wgt=False,
		# 		filter_tuple=filter_tuple_g,
		# 		filter_tuple_deltaU=(5, 5, 5),
		# 		plot_results=False,
		# 		plot_call_count=_inspect_call_count,
		# 		plot_output_dir=inspect_output_dir_g,
		# 	)
		# 	p_smooth_assembled = assemble_prediction(p_smooth_raw, **_assemble_kwargs)
		# 	p_local_assembled  = assemble_prediction(p_local_raw,  **_assemble_kwargs)
		# 	_plot_two_head_decomposition_multiZ(
		# 		p_smooth_assembled,
		# 		p_local_assembled,
		# 		_inspect_call_count,
		# 		inspect_output_dir_g,
		# 	)

		t0 = time.time()

		# CHANGED: use cached indices arrays for faster indexing
		if predict_ddUCorr_g:
			# change_in_deltap shape: (grid_z, grid_y, grid_x, 4)
			change_in_deltap_flat = change_in_deltap[indices_i, indices_j, indices_k, :]  # (n_cells, 4)
			ddp   = interpolate_fill_njit(change_in_deltap_flat[:, 0], vert_NPtoOF_array, weights_NPtoOF_array)
			delta_delta_U_x = interpolate_fill_njit(change_in_deltap_flat[:, 1], vert_NPtoOF_array, weights_NPtoOF_array)
			delta_delta_U_y = interpolate_fill_njit(change_in_deltap_flat[:, 2], vert_NPtoOF_array, weights_NPtoOF_array)
			delta_delta_U_z = interpolate_fill_njit(change_in_deltap_flat[:, 3], vert_NPtoOF_array, weights_NPtoOF_array)
		else:
			change_in_deltap = change_in_deltap[indices_i, indices_j, indices_k]
			ddp = interpolate_fill_njit(change_in_deltap, vert_NPtoOF_array, weights_NPtoOF_array)
		#p = deltaP_prev + change_in_deltap

		if verbose_g:
			print(f"Final Interpolation took: {time.time()-t0} s", flush=True)

		# CHANGED: vectorized split using np.split (faster than manual loop)
		if len(len_rankwise) != nprocs:
			raise ValueError(f"len_rankwise ({len(len_rankwise)}) does not match number of ranks ({nprocs})")
		if sum(len_rankwise) != len(ddp):
			raise ValueError(f"Sum of len_rankwise ({sum(len_rankwise)}) does not match length of ddp ({len(ddp)})")

		if predict_ddUCorr_g:
			# Stack into (n_cells, 4): [ddp, ddUx, ddUy, ddUz]
			combined = np.column_stack([ddp, delta_delta_U_x, delta_delta_U_y, delta_delta_U_z])
			output_rankwise = np.split(combined, np.cumsum(len_rankwise)[:-1])
		else:
			ddpML_rankwise = np.split(ddp, np.cumsum(len_rankwise)[:-1])
			output_rankwise = ddpML_rankwise

		if verbose_g:
			print(f"The whole python function took : {time.time()-t0_py_func} s")

	else:
		output_rankwise = None

	for output_rank_i in output_rankwise:
		if np.any(np.isnan(output_rank_i)):
			nan_count = np.sum(np.isnan(output_rank_i))
			if verbose_g:
				print(f"Number of NaN values in output_rankwise: {nan_count}")
			raise ValueError("Warning: NaN values detected in output_rankwise before scattering.")

	# This scatters the value to each worker
	result = comm.scatter(output_rankwise, root=0)

	if verbose_g:
		if np.any(np.isnan(result)):
			print(f"Warning: NaN values detected in result at rank {rank} after scattering.")

	if verbose_g:
		print(f"Process {rank} received object with shape {result.shape}", flush=True)

	# Enforce float64 dtype and check for NaNs/Infs before returning to C++
	result = np.asarray(result, dtype=np.float64)
	if not np.all(np.isfinite(result)):
		n_nan = np.isnan(result).sum()
		n_inf = np.isinf(result).sum()
		raise ValueError(f"Output array contains {n_nan} NaNs and {n_inf} Infs before returning to C++.")

	weighted_result = output_weight_factor_g * result

	if verbose_g and not predict_ddUCorr_g and rank == 0:
		# Physical check diagnostic: compare prediction vs previous (both are 1D arrays at mesh points)
		if 'change_in_deltap' in locals() and 'ddpPrev_interp_to_plot' in locals():
			mean_abs_prev = float(np.mean(np.abs(ddpPrev_interp_to_plot)))
			mean_abs_raw = float(np.mean(np.abs(change_in_deltap)))
			ratio_raw_to_prev = mean_abs_raw / (mean_abs_prev + 1e-16)
			rmse_raw_vs_prev = float(np.sqrt(np.mean((change_in_deltap - ddpPrev_interp_to_plot) ** 2)))

			print(
				"[final ddp physical check] rank 0 | "
				"mean|ddpPrev|={:.6e} | mean|raw_pred|={:.6e} | "
				"raw/prev={:.6f} | rmse(raw-prev)={:.6e}".format(
					mean_abs_prev,
					mean_abs_raw,
					ratio_raw_to_prev,
					rmse_raw_vs_prev,
				)
			)

	if verbose_g:
		print("max, min, mean of weighted_result at rank {}: {}, {}, {}".format(rank, weighted_result.max(), weighted_result.min(), weighted_result.mean()))
		print("first 3 values of weighted_result at rank {}: {}".format(rank, weighted_result[:3]))

	return weighted_result

if __name__ == '__main__':
    print('This is the Python module for DLPoissonFOam')