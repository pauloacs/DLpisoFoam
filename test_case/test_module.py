import h5py 

from python_module import init_func, py_func
import numpy as np
import matplotlib.pyplot as plt


array = np.load('test_data/array.npy')
top_boundary = np.load('test_data/top.npy')
obst_boundary = np.load('test_data/obst.npy')

def test_whole_module():

    plt.scatter(array[:,2],array[:,3], c = array[:,0])
    plt.savefig('ux.png')

    plt.scatter(array[:,2],array[:,3], c = array[:,1])
    plt.savefig('uy.png')

    init_func(array, top_boundary, obst_boundary)
    p = py_func(array)

    plt.scatter(array[:,2], array[:,3], c=p, cmap = 'jet')
    plt.savefig('p.png')

if __name__=="__main__":
    test_whole_module()

# from numba import njit
# @njit
# def index(array, item):
#    for idx, val in np.ndenumerate(array):
#        if val == item:
#            return idx
#    # If no item was found return None, other return types might be a problem due to
#    # numbas type inference.

#path = '/home/paulo/dataset_unsteadyCil_fu_bound.hdf5' #adjust path

#frame = 40
#hdf5_file = h5py.File(path, "r")
##data = hdf5_file["sim_data"][:1, frame-1:frame, ...]
#top_boundary = hdf5_file["top_bound"][0, frame, ...]
#obst_boundary = hdf5_file["obst_bound"][0, frame, ...]
#hdf5_file.close()

#indice_top = index(top_boundary[:,0] , -100.0 )[0]
#top_boundary = top_boundary[:indice_top,:]

#indice_obst = index(obst_boundary[:,0] , -100.0 )[0]
#obst_boundary = obst_boundary[:indice_obst,:]

#indice = index(data[0,0,:,0] , -100.0 )[0]
#array = data[0,0,:indice,:4]
#array[:,2:4] = data[0,0,:indice,3:5]
