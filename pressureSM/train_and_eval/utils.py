from numba import njit
import os
import shutil
import time
import h5py
import numpy as np

import matplotlib.pyplot as plt

import scipy.spatial.qhull as qhull
import matplotlib.path as mpltPath
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KDTree
from shapely.geometry import MultiPoint
from scipy.spatial import distance

def interp_weights(xyz, uvw, d=2):
    """
    Get interpolation weights and vertices using barycentric interpolation.

    This function calculates the interpolation weights and vertices for interpolating values from the original grid to the target grid.
    The interpolation is performed using Delaunay triangulation.

    Args:
        xyz (ndarray): Coordinates of the original grid.
        uvw (ndarray): Coordinates of the target grid.
        d (int, optional): Number of dimensions. Default is 2.

    Returns:
        ndarray: Vertices of the simplices that contain the target grid points.
        ndarray: Interpolation weights for each target grid point.
    """
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    wts = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
    valid = ~(simplex == -1)

    # Fill out-of-bounds points with Inverse-Distance Weighting
    if (~valid).any():
        tree = sklearn.neighbors.KDTree(xyz, leaf_size=40)
        nndist, nni = tree.query(np.array(uvw)[~valid], k=3)
        invalid = np.flatnonzero(~valid)
        vertices[invalid] = list(nni)
        wts[invalid] = list((1./np.maximum(nndist**2, 1e-6)) / (1./np.maximum(nndist**2, 1e-6)).sum(axis=-1)[:,None])
        
    return vertices, wts

def read_dataset(path, sim, time):
    """
    Reads dataset and splits it into the internal flow data (data) and boundary data.

    Args:
        path (str): Path to hdf5 dataset
        sim (int): Simulation number.
        time (int): Time frame.
    """
    with h5py.File(path, "r") as f:
        data = f["sim_data"][sim:sim+1, time:time+1, ...]
        top_boundary = f["top_bound"][sim:sim+1, time:time+1, ...]
        obst_boundary = f["obst_bound"][sim:sim+1, time:time+1, ...]

    return data, top_boundary, obst_boundary


#@njit
def interpolate_fill(values, vtx, wts, fill_value=np.nan):
    """
    Interpolate based on previously computed vertices (vtx) and weights (wts) and fill.

    Args:
        values (NDArray): Array of values to interpolate.
        vtx (NDArray): Array of interpolation vertices.
        wts (NDArray): Array of interpolation weights.
        fill_value (float): Value used to fill.
    
    Returns:
        NDArray: Interpolated values with fill_value for invalid weights.
    """
    ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    ret[np.any(wts < 0, axis=1)] = fill_value
    return ret


#@njit(nopython = True)  #much faster using numba.njit but is giving an error
def index(array, item):
    """
    Finds the index of the first element equal to item.

    Args:
        array (NDArray):
        item (float):
    """
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx
        # else:
        # 	return None
    # If no item was found return None, other return types might be a problem due to
    # numbas type inference.


def create_uniform_grid(x_min, x_max, y_min, y_max, delta):
    """
    Creates an uniform 2D grid (should envolve every cell of the mesh).

    Args:
        x_min (float): The variable name is self-explanatory.
        x_max (float): The variable name is self-explanatory.
        y_min (float): The variable name is self-explanatory.
        y_max (float): The variable name is self-explanatory.
    """
    X0 = np.linspace(x_min + delta/2 , x_max - delta/2 , num = int(round( (x_max - x_min)/delta )) )
    Y0 = np.linspace(y_min + delta/2 , y_max - delta/2 , num = int(round( (y_max - y_min)/delta )) )

    XX0, YY0 = np.meshgrid(X0,Y0)
    return XX0.flatten(), YY0.flatten()


def createGIF(n_sims, n_ts):
    
    ####################### TO CREATE A GIF WITH ALL THE FRAMES ###############################
    filenamesp = []

    for sim in range(5):
        for time in range(5):
            filenamesp.append(f'plots/p_pred_sim{sim}t{time}.png') #hardcoded to get the frames in order

    import imageio

    with imageio.get_writer('plots/p_movie.gif', mode='I', duration =0.5) as writer:
        for filename in filenamesp:
            image = imageio.imread(filename)
            writer.append_data(image)
    ######################## ---------------- //----------------- ###################

def plot_random_blocks(res_concat, y_array, x_array, sim, time, save_plots):
    """
    Plot 9 randomly sampled blocks for reference.

    Args:
        res_concat (ndarray): The array containing the predicted flow fields for each block.
        y_array (ndarray): The array containing the ground truth flow fields for each block.
        x_array (ndarray): The array containing the input flow fields for each block.
        sim (int): The simulation number.
        time (int): The time step number.
        save_plots (bool): Whether to save the plots.

    Returns:
        None
    """
    if save_plots:
        # plot blocks
        N = res_concat.shape[0]  # Number of blocks

        # Select 9 random indices
        random_indices = np.random.choice(N, size=9, replace=False)

        # Create the figure and axes for a 3x6 grid (3x3 for each side)
        fig, axes = plt.subplots(3, 6, figsize=(18, 12))

        # Add big titles for the left and right 3x3 grids with larger font size
        fig.text(0.25, 0.92, "SM Predictions", ha="center", fontsize=18, fontweight='bold')
        fig.text(0.75, 0.92, "CFD Predictions (Ground Truth)", ha="center", fontsize=18, fontweight='bold')

        for idx, i in enumerate(random_indices):
            row = idx // 3
            col = idx % 3

            # Plot SM predictions (left 3x3 grid)
            ax_sm = axes[row, col]
            masked_arr = np.ma.array(res_concat[i,:,:,0], mask=x_array[i,:,:,2]!=0)
            ax_sm.imshow(masked_arr, cmap='viridis')
            ax_sm.set_title(f"Block {i}/{N}", fontsize=12, fontweight='bold')
            ax_sm.axis("off")

            # Add border around each block for clearer distinction
            for _, spine in ax_sm.spines.items():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)

            # Plot CFD predictions (right 3x3 grid)
            ax_cfd = axes[row, col + 3]
            masked_arr = np.ma.array(y_array[i,:,:,0], mask=x_array[i,:,:,2]!=0)
            ax_cfd.imshow(masked_arr, cmap='viridis')
            ax_cfd.set_title(f"Block {i}/{N}", fontsize=12, fontweight='bold')
            ax_cfd.axis("off")

            # Add border around each block for clearer distinction
            for _, spine in ax_cfd.spines.items():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)

        # Adjust layout to make space for titles
        plt.tight_layout(rect=[0, 0, 1, 0.88])

        # Save the plot as an image file
        output_path = f"plots/sim{sim}/SM_vs_CFD_predictions_t{time}.png"  # Change the filename/path as needed
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

def compute_in_block_error(pred, true, flow_bool):

    true_masked = true[flow_bool]
    pred_masked = pred[flow_bool]

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

    ** Error in delta_p (blocks) **

        normVal  = {norm} Pa
        biasNorm = {BIAS_norm:.3f}%
        stdeNorm = {STDE_norm:.3f}%
        rmseNorm = {RMSE_norm:.3f}%
    """, flush = True)

    pred_minus_true_block = np.mean( (pred_masked  - true_masked )[mask_nan] )/norm
    pred_minus_true_squared_block = np.mean( (pred_masked  - true_masked )[mask_nan]**2 )/norm**2
    return pred_minus_true_block, pred_minus_true_squared_block


def domain_dist(i, top_boundary, obst_boundary, xy0):
    # boundaries indice
    indice_top = index(top_boundary[i,0,:,0] , -100.0 )[0]
    top = top_boundary[i,0,:indice_top,:]
    max_x, max_y, min_x, min_y = np.max(top[:,0]), np.max(top[:,1]) , np.min(top[:,0]) , np.min(top[:,1])

    is_inside_domain = ( xy0[:,0] <= max_x)  * ( xy0[:,0] >= min_x ) * ( xy0[:,1] <= max_y ) * ( xy0[:,1] >= min_y )

    # regular polygon for testing

    # # Matplotlib mplPath
    # path = mpltPath.Path(top_inlet_outlet)
    # is_inside_domain = path.contains_points(xy0)
    # print(is_inside_domain.shape)

    indice_obst = index(obst_boundary[i,0,:,0] , -100.0 )[0]
    obst = obst_boundary[i,0,:indice_obst,:]

    obst_points =  MultiPoint(obst)

    hull = obst_points.convex_hull       #only works for convex geometries
    hull_pts = hull.exterior.coords.xy    #have a code for any geometry . enven concave https://stackoverflow.com/questions/14263284/create-non-intersecting-polygon-passing-through-all-given-points/47410079
    hull_pts = np.c_[hull_pts[0], hull_pts[1]]

    path = mpltPath.Path(hull_pts)
    is_inside_obst = path.contains_points(xy0)

    domain_bool = is_inside_domain * ~is_inside_obst

    top = top[0:top.shape[0]:2,:]   #if this has too many values, using cdist can crash the memory since it needs to evaluate the distance between ~1M points with thousands of points of top
    obst = obst[0:obst.shape[0]:2,:]

    #print(top.shape)

    sdf = np.minimum( distance.cdist(xy0,obst).min(axis=1) , distance.cdist(xy0,top).min(axis=1) ) * domain_bool
    #print(np.max(distance.cdist(xy0,top).min(axis=1)))
    #print(np.max(sdf))

    return domain_bool, sdf


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
    
def normalize_PCA_data(input, output, standardization_method: str = "std"):
    if standardization_method == 'min_max':
        ## Option 2: Min-max scaling
        min_in = np.min(input, axis=0)
        max_in = np.max(input, axis=0)

        min_out = np.min(output, axis=0)
        max_out = np.max(output, axis=0)

        np.savez('min_max_values.npz', min_in=min_in, max_in=max_in, min_out=min_out, max_out=max_out)

        # Perform min-max scaling
        x = (input - min_in) / (max_in - min_in)
        y = (output - min_out) / (max_out - min_out)
        
    elif standardization_method == 'std':
        ## Option 1: Standardization
        mean_in = np.mean(input, axis=0)
        std_in = np.std(input, axis=0)

        mean_out = np.mean(output, axis=0)
        std_out = np.std(output, axis=0)

        np.savez('mean_std.npz', mean_in=mean_in, std_in=std_in, mean_out=mean_out, std_out=std_out)

        x = (input - mean_in) /std_in
        y = (output - mean_out) /std_out

    elif standardization_method == 'max_abs':
        # Option 3 - Old method
        max_abs_input_PCA = np.max(np.abs(input))
        max_abs_p_PCA = np.max(np.abs(output))
        print( max_abs_input_PCA, max_abs_p_PCA)

        np.savetxt('maxs_PCA', [max_abs_input_PCA, max_abs_p_PCA] )

        x = input/max_abs_input_PCA
        y = output/max_abs_p_PCA

    return x, y
        

## TF handling of training data

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def parse_single_image(input_parse, output_parse):
    #define the dictionary -- the structure -- of our single example
    data = {
    'height' : int64_feature(input_parse.shape[0]),
            'depth_x' : int64_feature(input_parse.shape[1]),
            'depth_y' : int64_feature(output_parse.shape[1]),
            'raw_input' : bytes_feature(tf.io.serialize_tensor(input_parse)),
            'output' : bytes_feature(tf.io.serialize_tensor(output_parse)),
        }

    #create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out

def write_images_to_tfr_short(input, output, filename:str="images"):
    filename= filename+".tfrecords"
    # Create a writer that'll store our data to disk
    writer = tf.io.TFRecordWriter(filename)
    count = 0

    for index in range(len(input)):
        #get the data we want to write
        current_input = input[index].astype('float32')
        current_output = output[index].astype('float32')

        out = parse_single_image(input_parse=current_input, output_parse=current_output)
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count

def parse_tfr_element(element):
    #use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'output' : tf.io.FixedLenFeature([], tf.string),
        'raw_input' : tf.io.FixedLenFeature([], tf.string),
        'depth_x':tf.io.FixedLenFeature([], tf.int64),
        'depth_y':tf.io.FixedLenFeature([], tf.int64)
        }

    content = tf.io.parse_single_example(element, data)

    height = content['height']
    depth_x = content['depth_x']
    depth_y = content['depth_y']
    output = content['output']
    raw_input = content['raw_input']
        
    #get our 'feature'-- our image -- and reshape it appropriately

    input_out= tf.io.parse_tensor(raw_input, out_type=tf.float32)
    output_out = tf.io.parse_tensor(output, out_type=tf.float32)

    return ( input_out , output_out)

def Callback_EarlyStopping(LossList, min_delta=0.1, patience=20):
    #No early stopping for 2*patience epochs
    if len(LossList)//patience < 2 :
        return False
    #Mean loss for last patience epochs and second-last patience epochs
    mean_previous = np.mean(LossList[::-1][patience:2*patience]) #second-last
    mean_recent = np.mean(LossList[::-1][:patience]) #last
    #you can use relative or absolute change
    delta_abs = np.abs(mean_recent - mean_previous) #abs change
    delta_abs = np.abs(delta_abs / mean_previous)  # relative change
    if delta_abs < min_delta :
        print("*CB_ES* Loss didn't change much from last %d epochs"%(patience))
        print("*CB_ES* Percent change in loss value:", delta_abs*1e2)
        return True
    else:
        return False

def load_dataset_tf(filename, batch_size, buffer_size):
    #create the dataset
    dataset = tf.data.TFRecordDataset(filename)

    #pass every single feature through our mapping function
    dataset = dataset.map(parse_tfr_element)

    dataset = dataset.shuffle(buffer_size=buffer_size )
    #epoch = tf.data.Dataset.range(epoch_num)
    dataset = dataset.batch(batch_size)

    return dataset

def define_model_arch(model_architecture: str) -> tuple[int, list]:

  if model_architecture == 'MLP_small':
    n_layers = 3
    width = [512]*3
  elif model_architecture == 'MLP_big':
    n_layers = 7
    width = [256] + [512]*5 + [256]
  elif model_architecture == 'MLP_huge':
    n_layers = 12
    width = [256] + [512]*10 + [256]
  elif model_architecture == 'MLP_huger':
    n_layers = 20
    width = [256] + [512]*18 + [256]
  elif model_architecture == 'MLP_small_unet':
    n_layers = 9
    width = [512, 256, 128, 64, 32, 64, 128, 256, 512]
  elif model_architecture == 'conv1D':
    n_layers = 7
    width = [128, 64, 32, 16, 32, 64, 128]
  elif model_architecture == 'MLP_attention':
    n_layers = 3
    width = [512]*3
  else:
    raise ValueError('Invalid NN model type')

  return n_layers, width