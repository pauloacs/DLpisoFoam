#f = open('python_log_file','w')
# f.write('Starting python module from OpenFOAM')
# f.close()

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

from surrogate_models.u_to_p.utils import *


def load_pca_and_NN(ipca_input_fn, ipca_output_fn, maxs_fn, max_PCA_fn, weights_fn):
	print('Loading the PCA mapping')

	pcainput = pk.load(open(ipca_input_fn, 'rb'))
	pcap = pk.load(open(ipca_output_fn, 'rb'))

	maxs = np.loadtxt(maxs_fn)
	maxs_PCA = np.loadtxt(max_PCA_fn)

	global max_abs_Ux, max_abs_Uy, max_abs_dist, max_abs_p
	max_abs_Ux, max_abs_Uy, max_abs_dist, max_abs_p = maxs[0], maxs[1], maxs[2], maxs[3]
	global max_abs_input_PCA, max_abs_p_PCA
	max_abs_input_PCA, max_abs_p_PCA = maxs_PCA[0], maxs_PCA[1]

	PC_p = int(np.argmax(pcap.explained_variance_ratio_.cumsum() > 0.95))
	PC_input = int(np.argmax(pcainput.explained_variance_ratio_.cumsum() > 0.995))

	global comp_p, pca_mean_p, comp_input, pca_mean_input, model
	comp_p = pcap.components_[:PC_p, :]
	pca_mean_p = pcap.mean_
	comp_input = pcainput.components_[:PC_input,:]
	pca_mean_input = pcainput.mean_

	print('Initializing NN')
	model = DENSE_PCA((PC_input), PC_p)
	model.load_weights(weights_fn)

	global comm, rank, nprocs
	print('Initializing MPI communication in Python')
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	nprocs = comm.Get_size()
			


def init_func(array, top_boundary, obst_boundary):
	"""
	Initialization function. It is called at the beggining of a simulation to compute everything that is static,
	including interpolation weights and vertices

	Args:
		array: Ux, Uy, and coordinates at each mesh cell center.
		top_boundary: 
		obst_boundary:
	"""
	print('Running init function... This may take a while! ')

	array_global = comm.gather(array, root = 0)
	top_global = comm.gather(top_boundary, root = 0)
	obst_global = comm.gather(obst_boundary, root = 0)

	global len_rankwise

	len_rankwise = comm.gather(array.shape[0], root = 0)

	if rank == 0:

		array_concat = np.concatenate(array_global)
		top = np.concatenate(top_global)
		obst = np.concatenate(obst_global)

		#np.save('top.npy', top)
		#np.save('obst.npy', obst)
		#np.save('array.npy', array_concat)
		
		global indices, sdfunct, vert_OFtoNP, weights_OFtoNP, vert_NPtoOF, weights_NPtoOF, grid_shape_y, grid_shape_x

		# Uniform grid resolution
		delta = 5e-3 

		x_min = round(np.min(array_concat[...,2]),2)
		x_max = round(np.max(array_concat[...,2]),2)

		y_min = round(np.min(array_concat[...,3]),2)
		y_max = round(np.max(array_concat[...,3]),2)

		X0, Y0 = create_uniform_grid(x_min, x_max, y_min, y_max, delta)

		xy0 = np.concatenate((np.expand_dims(X0, axis=1),np.expand_dims(Y0, axis=1)), axis=-1)
		points = array_concat[...,2:4] #coordinates


		#print( 'Calculating verts and weights' )
		vert_OFtoNP, weights_OFtoNP = interp_weights(points, xy0) #takes ~100% of the interpolation cost and it's only done once for each different mesh/simulation case
		vert_NPtoOF, weights_NPtoOF = interp_weights(xy0, points) #takes ~100% of the interpolation cost and it's only done once for each different mesh/simulation case
	
		#print( 'Calculating domain bool' )
		domain_bool, sdf = domain_dist(top, obst, xy0)

		grid_shape_y = int(round((y_max-y_min)/delta)) 
		grid_shape_x = int(round((x_max-x_min)/delta))
		block_size = 128

		x0 = np.min(X0)
		y0 = np.min(Y0)
		dx = delta
		dy = delta

		indices= np.empty((X0.shape[0],2))
		obst_bool = np.zeros((grid_shape_y,grid_shape_x,1))
		sdfunct = np.zeros((grid_shape_y,grid_shape_x,1))

		#to compute bool 
		ux = array_concat[...,0:1] #values
		ux_interp = interpolate_fill(ux, vert_OFtoNP, weights_OFtoNP) 

		for (step, x_y) in enumerate(np.c_[X0, Y0]):  
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

def py_func(array_in, verbose=True):
	"""
	Method called at each simulation time step to compute the pressure field based on an input velocity field.

	Args:
		array_in: 
	"""

	# if rank ==0:
	# 	print(memory())

	# Gathering all the inputs in 1 thread

	array_global = comm.gather(array_in, root = 0)

	if rank == 0: #running all calculations at rank 0 

		t0_py_func = time.time()

		array = np.concatenate(array_global)

		t0 = time.time()
		p_prev = array[...,4]

		#np.save('array.npy', array)
		U_max_norm = np.max( np.sqrt( np.square(array[...,0:1]) + np.square(array[...,1:2]) ) )

		Ux_adim = array[...,0:1]/U_max_norm 
		Uy_adim = array[...,1:2]/U_max_norm 

		t1 = time.time()
		if verbose: 
			print( "Data pre-processing:" + str(t1-t0) + " s")

		t0 = time.time()

		Ux_interp = interpolate(Ux_adim, vert_OFtoNP, weights_OFtoNP)
		Uy_interp = interpolate(Uy_adim, vert_OFtoNP, weights_OFtoNP)

		t1 = time.time()
		if verbose:
			print( "1st interpolation took:" + str(t1-t0) + " s")

		t0 = time.time()
		grid = np.zeros(shape=(1, grid_shape_y, grid_shape_x, 3))

		grid[0,:,:,0:1][tuple(indices.T)] = Ux_interp.reshape(Ux_interp.shape[0],1)/max_abs_Ux
		grid[0,:,:,1:2][tuple(indices.T)] = Uy_interp.reshape(Uy_interp.shape[0],1)/max_abs_Uy
		grid[0,:,:,2:3] = sdfunct

		t1 = time.time()
		if verbose:
			print( "2nd interpolation took:" + str(t1-t0) + " s")

		# Setting any nan value to 0 to avoid issues
		grid[np.isnan(grid)] = 0

		x_list = []
		obst_list = []
		indices_list = []

		shape = 128
		avance = int(0.1*shape)

		n_x = int((grid.shape[2]-shape)/(shape - avance ))  
		n_y = int((grid.shape[1]-shape)/(shape - avance ))


		t0 = time.time()

		# Sampling blocks of size [shape X shape] from the input fields (Ux, Uy and sdf)
		# In the indices_list, the indices corresponding to each sampled block is stored to enable domain reconstruction later.
		for i in range (n_y + 2):
				for j in range (n_x +1):

					if i == (n_y + 1 ):
						x_list.append(grid[0:1, (grid.shape[1]-shape):grid.shape[1] , ((grid.shape[2]-shape)-j*shape+j*avance):(grid.shape[2]-j*shape +j*avance) ,0:3])
						indices_list.append([i, n_x - j])

					else:
						x_list.append(grid[0:1,(i*shape - i*avance):(shape*(i+1) - i*avance),((grid.shape[2]-shape)-j*shape+j*avance):(grid.shape[2]-j*shape +j*avance),0:3])
						indices_list.append([i, n_x - j]) #will be used to rearrange the output

					if ( j ==  n_x ) and ( i == (n_y+1) ): #last one
						x_list.append(grid[0:1, (grid.shape[1]-shape):grid.shape[1] , 0:shape ,0:3])
						indices_list.append([i,-1])

					elif j == n_x :
						x_list.append(grid[0:1,i*shape - i*avance :shape*(i+1) -i*avance , 0:shape ,0:3])
						indices_list.append([i,-1])


		x_array = np.concatenate(x_list)

		t1 = time.time()
		#print( "Data extraction loop took:" + str(t1-t0) + " s")

		t0 = time.time()
		N = x_array.shape[0]
		features = x_array.shape[3]

		x_array_flat = x_array.reshape((N, x_array.shape[1]*x_array.shape[2], features ))

		input_flat = x_array_flat.reshape((x_array_flat.shape[0],-1))
		#input_transformed = pcainput.transform(input_flat)[:,:PC_input]

		input_transformed = np.dot(input_flat - pca_mean_input, comp_input.T)

		x_input = input_transformed/max_abs_input_PCA
		x_input = np.array(x_input)

		t1 = time.time()
		#print( "PCA transform : " + str(t1-t0) + " s")

		result_array = np.empty(grid[...,0:1].shape)
		t0 = time.time()

		# Calling the NN to predict the principal components (PC) of the pressure field:
		# PC_input -> PC_p (if necessary could be done in batches)
		res_concat = np.array(model(x_input)) 
		t1 = time.time()
		#print( "Model prediction time : " + str(t1-t0) + " s")

		# PCA inverse transformation:
		# PC_p -> p
		t0 = time.time()
		res_flat_inv = np.dot(res_concat*max_abs_p_PCA, comp_p) + pca_mean_p	
		res_concat = res_flat_inv.reshape((res_concat.shape[0], shape, shape, 1))
		t1 = time.time()
		#print( "PCA inverse transform : " + str(t1-t0) + " s")

		## Domain reassembly method
		flow_bool_ant = np.ones((shape,shape))
		BC_up = 0
		BC_ant = 0
		BC_alter = 0

		BC_ups = np.zeros(n_x+1)

		t0 = time.time()

		for i in range(len(x_list)):

				idx = indices_list[i]
				flow_bool = x_array[i,:,:,2]
				res = res_concat[i,:,:,0]

				if idx[0] == 0: 
					if idx[1] == n_x :

						BC_coor = np.mean(res[:,(shape-avance):shape][flow_bool[:,(shape-avance):shape]!=0]) - BC_up
						res -= BC_coor
						BC_ups[idx[1]] = np.mean(res[(shape-avance):shape,(shape-avance):shape][flow_bool[(shape-avance):shape,(shape-avance):shape] !=0])	

					elif idx[1] == -1:
						p_j = (grid.shape[2]-shape)-n_x*shape+n_x*avance
						BC_coor = np.mean(res[:, p_j:p_j + avance][flow_bool[:, p_j:p_j + avance] !=0] ) - BC_ant_0 #middle ones are corrected by the right side
						res -= BC_coor
						BC_up_ = np.mean(res[(shape-avance):shape, p_j:p_j + avance][flow_bool[(shape-avance):shape, p_j:p_j + avance] !=0] ) #equivale a BC_ups[idx[1]==-1]
					else:
						BC_coor = np.mean(res[:,(shape-avance):shape][flow_bool[:,(shape-avance):shape] !=0] ) - BC_ant_0 
						res -= BC_coor
						BC_ups[idx[1]] = np.mean(res[(shape-avance):shape,:][flow_bool[(shape-avance):shape,:] !=0])	
					BC_ant_0 =  np.mean(res[:,0:avance][flow_bool[:,0:avance] !=0]) 	

				elif idx[0] == n_y+1 : 
					if idx[1] == -1: 
				
						p = grid.shape[1] - (shape*(n_y+1) - n_y*avance)
						p_j = (grid.shape[2]-shape)-n_x*shape+n_x*avance
						BC_coor = np.mean(res[shape - p -avance: shape - p , p_j: p_j + avance][flow_bool[ shape - p -avance: shape - p , p_j: p_j + avance] !=0] ) - BC_up_
						res -= BC_coor
					else: 

						p = grid.shape[1] - (shape*(n_y+1) - n_y*avance)
						if np.isnan(BC_ups[idx[1]]):
							BC_coor = np.mean(res[:,shape-avance:shape][flow_bool[:,shape-avance:shape] !=0]) - BC_alter
						else:
							BC_coor = np.mean(res[shape - p -avance: shape - p,:][flow_bool[shape - p -avance: shape - p,:] !=0]) - BC_ups[idx[1]]

						res -= BC_coor	

				else:

					if idx[1] == -1:

						p_j = (grid.shape[2]-shape)-n_x*shape+n_x*avance
						BC_coor = np.mean(res[0:avance,p_j: p_j + avance ][flow_bool[0:avance,p_j: p_j + avance ]!=0]) - BC_up_
						res -= BC_coor
						BC_up_ = np.mean(res[(shape-avance):shape, p_j: p_j + avance])

					else:

						if np.isnan(BC_ups[idx[1]]):
							BC_coor = np.mean(res[: ,shape-avance:shape][flow_bool[: ,shape-avance:shape] !=0]) - BC_alter
						else:
							BC_coor = np.mean(res[0:avance,:][flow_bool[0:avance,:]!=0]) - BC_ups[idx[1]]
						res -= BC_coor 
						BC_ups[idx[1]] = np.mean(res[(shape-avance):shape,:][flow_bool[(shape-avance):shape,:] !=0])	
						

				#BC alternative: when it is not possible to apply the correction based on the right side 
				# when it is not possible to correct based on the block above
				BC_alter = np.mean(res[:,0:avance][flow_bool[:,0:avance] !=0]) 

				if idx == [n_y +1, -1]:
					result_array[0,(grid.shape[1]-(shape-avance)):grid.shape[1] , 0:(grid.shape[2] - (n_x+1)*(shape-avance) -avance) ,0] = res[avance:shape , 0:grid.shape[2] - (n_x+1)*(shape-avance) -avance]

				elif idx[1] == -1:
					result_array[0,(idx[0]*shape - idx[0]*avance):(1+idx[0])*shape - idx[0]*avance, 0:shape,0] = res


				elif idx[0] == (n_y + 1):
					j = n_x - idx[1]
					result_array[0,(grid.shape[1]-(shape-avance)):grid.shape[1], grid.shape[2] -shape - j*(shape-avance) : grid.shape[2] - j*(shape-avance) ,0] = res[avance:shape,:]

				else:
					j = n_x - idx[1]
					result_array[0,(idx[0]*shape - idx[0]*avance):(1+idx[0])*shape - idx[0]*avance, grid.shape[2] -shape - j*(shape-avance) : grid.shape[2] - j*(shape-avance) ,0] = res

		t1 = time.time()
		if verbose:
			print( "Reassembly algorithm took:" + str(t1-t0) + " s")

		# Ensuring the constant pressure boundary condition to be ensured at the right-most cell center
		# instead of doing that at the boundary patch, brings bias to the domain, removing it below:
		result_array -= np.mean( 3* result_array[0,:,-1,0] - result_array[0,:,-2,0] )/3
		result_array = result_array[0,:,:,0]

		t0 = time.time()
		# Apply Gaussian filter to correct the attained pressure field (and remove artifacts) (OPTIONAL)
		#result_array = ndimage.gaussian_filter(result_array, sigma=(5, 5), order=0)
		t1 = time.time()
		if verbose:
			print( "Applying Gaussian filter took:" + str(t1-t0) + " s")

		# Rearrange the prediction into a 1D array that it can be sent to OF
		p_adim_unif = result_array[tuple(indices.T)]

		t0 = time.time()
		# Interpolation into the orginal grid
		# Takes virtually no time because "vert" and "weigths" where already calculated on the init_func
		p_interp = interpolate_fill(p_adim_unif, vert_NPtoOF, weights_NPtoOF)
		#p_interp[np.isnan(p_interp)] = 0
		t1 = time.time()
		if verbose:
			print( "Final Interpolation took:" + str(t1-t0) + " s")

		# Finally redimensionalizing the predicted pressure field
		p = p_interp * max_abs_p * pow(U_max_norm, 2.0)

		# Using the last time step pressure field for the near wall locations
		# Because we know that the model underperforms at the grid elements near walls
		sdf_mesh = interpolate_fill(sdfunct[:,:,0], vert_NPtoOF, weights_NPtoOF)
		#p[sdf_mesh < 0.05] = p_prev[sdf_mesh < 0.05]
		#p[np.isnan(p_interp)] = p_prev[np.isnan(p_interp)]

		t1_py_func = time.time()
		if verbose:
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
