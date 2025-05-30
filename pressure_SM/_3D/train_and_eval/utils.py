from numba import njit
import os
import h5py
import numpy as np
import pyvista as pv

import matplotlib.pyplot as plt

import scipy.spatial.qhull as qhull
import matplotlib.path as mpltPath
import tensorflow as tf
import sklearn
from shapely.geometry import MultiPoint
from scipy.spatial import distance
from scipy.spatial import ConvexHull, Delaunay, cKDTree

from matplotlib import cm
from matplotlib.colors import Normalize
from pyDOE import lhs
import pickle as pk
import tables

def interp_weights(xyz, uvw, d=3, interp_method='IDW'):
    """
    Get interpolation weights and vertices using barycentric interpolation.

    This function calculates the interpolation weights and vertices for interpolating values from the original grid to the target grid.
    The interpolation is performed using Delaunay triangulation.

    Args:
        xyz (ndarray): Coordinates of the original grid.
        uvw (ndarray): Coordinates of the target grid.
        d (int, optional): Number of dimensions. Default is 3.

    Returns:
        ndarray: Vertices of the simplices that contain the target grid points.
        ndarray: Interpolation weights for each target grid point.
    """
    # For 3D data, baricentric interpolation is very slow - so IDW is the default

    if interp_method == "IDW":
        tree = sklearn.neighbors.KDTree(xyz, leaf_size=40)
        nndist, nni = tree.query(np.array(uvw), k=3)
        vertices = list(nni)
        wts = list((1./np.maximum(nndist**2, 1e-6)) / (1./np.maximum(nndist**2, 1e-6)).sum(axis=-1)[:,None])

    elif interp_method == "barycentric":
        tri = qhull.Delaunay(xyz)
        simplex = tri.find_simplex(uvw)
        vertices = np.take(tri.simplices, simplex, axis=0)
        temp = np.take(tri.transform, simplex, axis=0)
        delta = uvw - temp[:, d]
        bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
        wts = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
        valid = ~(simplex == -1)

        import dask.array as da
        from scipy.spatial import Delaunay

        # Convert input arrays to dask arrays
        xyz_dask = da.from_array(xyz, chunks='auto')
        uvw_dask = da.from_array(uvw, chunks='auto')

        # Perform Delaunay triangulation
        tri = Delaunay(xyz)

        # Find the simplex containing each point in uvw
        simplex = da.map_blocks(tri.find_simplex, uvw_dask, dtype=int)
        vertices = da.map_blocks(np.take, tri.simplices, simplex, axis=0, dtype=int)
        temp = da.map_blocks(np.take, tri.transform, simplex, axis=0, dtype=float)
        delta = uvw_dask - temp[:, d]
        bary = da.einsum('njk,nk->nj', temp[:, :d, :], delta)
        wts = da.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
        valid = ~(simplex == -1)

        # Compute the results
        vertices = vertices.compute()
        wts = wts.compute()
        valid = valid.compute()

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
        data = np.array(f["sim_data"][sim, time, ...], dtype='float32')
        obst_boundary = np.array(f["obst_bound"][sim, time, ...], dtype='float32')
        y_bot_boundary = np.array(f["y_bot_bound"][sim, time, ...], dtype='float32')
        z_bot_boundary = np.array(f["z_bot_bound"][sim, time, ...], dtype='float32')
        y_top_boundary = np.array(f["y_top_bound"][sim, time, ...], dtype='float32')
        z_top_boundary = np.array(f["z_top_bound"][sim, time, ...], dtype='float32')

    return data, obst_boundary, y_bot_boundary, z_bot_boundary, y_top_boundary, z_top_boundary


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
    ret[np.any(np.array(wts) < 0, axis=1)] = fill_value
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


def create_uniform_grid(x_min, x_max, y_min, y_max, z_min, z_max, delta):
    """
    Creates an uniform 2D grid (should envolve every cell of the mesh).

    """
    X0 = np.linspace(x_min + delta/2 , x_max - delta/2 , num = int(round( (x_max - x_min)/delta )) )
    Y0 = np.linspace(y_min + delta/2 , y_max - delta/2 , num = int(round( (y_max - y_min)/delta )) )
    Z0 = np.linspace(z_min + delta/2 , z_max - delta/2 , num = int(round( (z_max - z_min)/delta )) )

    XX0, YY0, ZZ0 = np.meshgrid(X0,Y0,Z0)
    return XX0.flatten(), YY0.flatten(), ZZ0.flatten()


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

# NOT WORKING .... 
def plot_random_blocks_3d_render(res_concat, y_array, x_array, sim, time, save_plots):
    """
    Plot 9 randomly sampled blocks for reference.

    Currently plotting slices. 
    Volumetric rendering is not working ... don't know why yet ...

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
            #
            # Plot SM predictions (left 3x3 grid)
            ax_sm = axes[row, col]
            p_sm = res_concat[i,:,:,:,0]
            grid_sm = pv.ImageData(dimensions=p_sm.shape)
            grid_sm['Pressure'] = p_sm.flatten(order='F')
            slices_sm = grid_sm.slice_orthogonal()
            pl_sm = pv.Plotter(off_screen=True)
            pl_sm.add_mesh(slices_sm, cmap='viridis', show_edges=False)
            pl_sm.add_title(f"Block {i} - Slices", font_size=18)
            screenshot_sm = pl_sm.screenshot()
            ax_sm.imshow(screenshot_sm)
            ax_sm.set_title(f"Block {i}/{N}", fontsize=12, fontweight='bold')
            ax_sm.axis("off")
            #
            # # Add border around each block for clearer distinction
            # for _, spine in ax_sm.spines.items():
            #     spine.set_edgecolor('black')
            #     spine.set_linewidth(2)
            #
            # Plot CFD predictions (right 3x3 grid)
            ax_cfd = axes[row, col + 3]
            p_cfd = y_array[i,:,:,:,0]
            grid_cfd = pv.ImageData(dimensions=p_sm.shape)
            grid_cfd['Pressure'] = p_cfd.flatten(order='F')
            slices_cfd = grid_cfd.slice_orthogonal()
            pl_cfd = pv.Plotter(off_screen=True)
            pl_cfd.add_mesh(slices_cfd, cmap='viridis', show_edges=False)
            pl_cfd.add_title(f"Block {i} - Slices", font_size=18)
            screenshot_cfd = pl_cfd.screenshot()
            ax_cfd.imshow(screenshot_cfd)
            ax_cfd.set_title(f"Block {i}/{N}", fontsize=12, fontweight='bold')
            ax_cfd.axis("off")
            #
            # # Add border around each block for clearer distinction
            # for _, spine in ax_cfd.spines.items():
            #     spine.set_edgecolor('black')
            #     spine.set_linewidth(2)

            # Adjust layout to make space for titles
            plt.tight_layout(rect=[0, 0, 1, 0.88])

            # Save the plot as an image file
            output_path = f"plots/sim{sim}/SM_vs_CFD_predictions_t{time}.png"  # Change the filename/path as needed
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)


# NOT WORKING .... 
def plot_random_blocks(res_concat, y_array, x_array, sim, time, save_plots):
    """
    Plot 9 randomly sampled blocks for reference.

    Currently plotting slices. 
    Volumetric rendering is not working ... don't know why yet ...

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
            #
            # Plot SM predictions (left 3x3 grid)
            ax_sm = axes[row, col]
            p_sm = res_concat[i,:,:,:,0]
            grid_sm = pv.ImageData(dimensions=p_sm.shape)
            grid_sm['Pressure'] = p_sm.flatten(order='F')
            slices_sm = grid_sm.slice_orthogonal()
            pl_sm = pv.Plotter(off_screen=True)
            pl_sm.add_mesh(slices_sm, cmap='viridis', show_edges=False)
            pl_sm.add_title(f"Block {i} - Slices", font_size=18)
            screenshot_sm = pl_sm.screenshot()
            ax_sm.imshow(screenshot_sm)
            ax_sm.set_title(f"Block {i}/{N}", fontsize=12, fontweight='bold')
            ax_sm.axis("off")
            #
            # # Add border around each block for clearer distinction
            # for _, spine in ax_sm.spines.items():
            #     spine.set_edgecolor('black')
            #     spine.set_linewidth(2)
            #
            # Plot CFD predictions (right 3x3 grid)
            ax_cfd = axes[row, col + 3]
            p_cfd = y_array[i,:,:,:,0]
            grid_cfd = pv.ImageData(dimensions=p_sm.shape)
            grid_cfd['Pressure'] = p_cfd.flatten(order='F')
            slices_cfd = grid_cfd.slice_orthogonal()
            pl_cfd = pv.Plotter(off_screen=True)
            pl_cfd.add_mesh(slices_cfd, cmap='viridis', show_edges=False)
            pl_cfd.add_title(f"Block {i} - Slices", font_size=18)
            screenshot_cfd = pl_cfd.screenshot()
            ax_cfd.imshow(screenshot_cfd)
            ax_cfd.set_title(f"Block {i}/{N}", fontsize=12, fontweight='bold')
            ax_cfd.axis("off")
            #
            # # Add border around each block for clearer distinction
            # for _, spine in ax_cfd.spines.items():
            #     spine.set_edgecolor('black')
            #     spine.set_linewidth(2)

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


def domain_dist(boundaries_list, xyz0, grid_res, find_limited_index=True):
    
    obst_boundary, y_bot_boundary, z_bot_boundary, y_top_boundary, z_top_boundary = boundaries_list

    # Always compute the domain limits from the boundaries
    max_x, max_y, min_x, min_y = (
        np.max(z_top_boundary[:,0]) + grid_res,
        np.max(z_top_boundary[:,1]) + grid_res,
        np.min(z_top_boundary[:,0]) - grid_res,
        np.min(z_top_boundary[:,1]) - grid_res
    )
    max_z, min_z = np.max(y_top_boundary[:,2]), np.min(y_top_boundary[:,2])

    if find_limited_index:
        indice_z_top = index(z_top_boundary[:,0], -100.0)[0]
        z_top = z_top_boundary[:indice_z_top, :]
        indice_y_top = index(y_top_boundary[:,0], -100.0)[0]
        y_top = y_top_boundary[:indice_y_top, :]
        indice_y_bot = index(y_bot_boundary[:,0], -100.0)[0]
        y_bot = y_bot_boundary[:indice_y_bot, :]
        indice_z_bot = index(z_bot_boundary[:,0], -100.0)[0]
        z_bot = z_bot_boundary[:indice_z_bot, :]
        indice_obst = index(obst_boundary[:,0] , -100.0 )[0]
        obst = obst_boundary[:indice_obst,:]
    else:
        z_top = z_top_boundary
        y_top = y_top_boundary
        y_bot = y_bot_boundary
        z_bot = z_bot_boundary
        obst = obst_boundary

    # Since most times the outer domain is a parallelogram
    # using this simplified approach
    is_inside_domain = (
        (xyz0[:, 0] <= max_x) &
        (xyz0[:, 0] >= min_x) &
        (xyz0[:, 1] <= max_y) &
        (xyz0[:, 1] >= min_y) &
        (xyz0[:, 2] <= max_z) &
        (xyz0[:, 2] >= min_z)
    )

    # Alternatively, the following could be used:
    # regular polygon for testing
    # # Matplotlib mplPath
    # path = mpltPath.Path(top_inlet_outlet)
    # is_inside_domain = path.contains_points(xy0)
    # print(is_inside_domain.shape)


    
    # obst_points =  MultiPoint(obst)
    # # Convex hull of obstacle - only works for convex geometries
    # #For any geometry (enven concave), check
    # # https://stackoverflow.com/questions/14263284/create-non-intersecting-polygon-passing-through-all-given-points/47410079
    # hull = obst_points.convex_hull
    # hull_pts = np.array(hull.exterior.coords)

    # path = mpltPath.Path(hull_pts)
    # is_inside_obst = path.contains_points(xyz0)

    # Check points inside the obstacle
    hull = ConvexHull(obst)
    delaunay = Delaunay(hull.points[hull.vertices])
    is_inside_obst = delaunay.find_simplex(xyz0) >= 0

    # Flow domain boolean
    domain_bool = is_inside_domain * ~is_inside_obst

    # if this has too many values, using cdist can crash the memory
    # since it needs to evaluate the distance between ~1M points with thousands of points of top
    # increasing the step lowers the cost
    step = 1
    z_top = z_top[0:z_top.shape[0]:step,:]
    z_bot = z_bot[0:z_bot.shape[0]:step,:]
    y_top = y_top[0:y_top.shape[0]:step,:]
    z_bot = z_bot[0:z_bot.shape[0]:step,:]
    obst = obst[0:obst.shape[0]:step,:]

    # sdf = np.minimum(
    #     distance.cdist(xyz0,obst).min(axis=1),
    #     distance.cdist(xyz0,z_top).min(axis=1),
    #     distance.cdist(xyz0,z_bot).min(axis=1),
    #     distance.cdist(xyz0,y_top).min(axis=1),
    #     distance.cdist(xyz0,y_bot).min(axis=1),
    #     ) * domain_bool

    # return domain_bool, sdf

    # Pre-build KD trees for each surface
    obst_tree = cKDTree(obst)
    z_top_tree = cKDTree(z_top)
    z_bot_tree = cKDTree(z_bot)
    y_top_tree = cKDTree(y_top)
    y_bot_tree = cKDTree(y_bot)

    # Query each tree for the nearest distances to xyz0
    obst_dist, _ = obst_tree.query(xyz0, k=1)
    z_top_dist, _ = z_top_tree.query(xyz0, k=1)
    z_bot_dist, _ = z_bot_tree.query(xyz0, k=1)
    y_top_dist, _ = y_top_tree.query(xyz0, k=1)
    y_bot_dist, _ = y_bot_tree.query(xyz0, k=1)

    # Calculate signed distance field (sdf)
    sdf = np.minimum.reduce([obst_dist, z_top_dist, z_bot_dist, y_top_dist, y_bot_dist]) * domain_bool

    return domain_bool, sdf

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
    
def normalize_feature_data(input, output, standardization_method: str = "std"):
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

        ## stds are inf

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

    model_architecture = model_architecture.lower()
    match model_architecture:
        case 'mlp_small':
            n_layers = 3
            width = [512]*3
        case 'mlp_big':
            n_layers = 7
            width = [256] + [512]*5 + [256]
        case 'mlp_huge':
            n_layers = 12
            width = [256] + [512]*10 + [256]
        case 'mlp_small_unet':
            n_layers = 9
            width = [512, 256, 128, 64, 32, 64, 128, 256, 512]
        case 'conv1d':
            n_layers = 7
            width = [128, 64, 32, 16, 32, 64, 128]
        case 'mlp_attention':
            n_layers = 3
            width = [512]*3
        case 'mlp_medium':
            n_layers = 5
            width = [256, 512, 512, 512, 256]
        case 'gnn':
            n_layers = 6
            width = [128, 256, 256, 256, 128, 64]
        case 'fno3d':
            n_layers = 4
            width = [64, 128, 128, 64]
        case _:
            raise ValueError('Invalid NN model type')

    return n_layers, width


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
from scipy import ndimage

def get_facecolors(data_slice, norm, cmap):
    """Map data to colors, ensuring NaNs are transparent."""
    normed_data = norm(data_slice)  # Normalize data
    rgba_colors = cmap(normed_data)  # Map to colormap
    rgba_colors[np.isnan(data_slice)] = [0, 0, 0, 0]  # Make NaNs fully transparent
    return rgba_colors

def plot_delta_p_comparison(cfd_results, field_deltap, no_flow_bool, slices_indices=[5, 50, 95], fig_path=None):
    # Mask the error field
    error = np.abs((field_deltap - cfd_results) / (np.max(cfd_results) - np.min(cfd_results)) * 100)
    masked_error = np.where(no_flow_bool, np.nan, error)

    # Masking the predicted delta_p field
    masked_deltap = np.where(no_flow_bool, np.nan, field_deltap)

    # Mask the CFD results
    masked_cfd = np.where(no_flow_bool, np.nan, cfd_results)

    # Create figure and axes
    fig = plt.figure(figsize=(20, 8))
    ax_deltap = fig.add_subplot(131, projection='3d')
    ax_cfd = fig.add_subplot(132, projection='3d')
    ax_error = fig.add_subplot(133, projection='3d')

    # Create meshgrid
    X, Y = np.meshgrid(np.arange(masked_cfd.shape[2]), np.arange(masked_cfd.shape[1]))

    # Set colormap and normalization
    cmap = cm.viridis
    vmin = min(np.nanmin(masked_cfd), np.nanmin(masked_deltap))
    vmax = max(np.nanmax(masked_cfd), np.nanmax(masked_deltap))
    norm = Normalize(vmin=vmin, vmax=vmax)
    error_norm = Normalize(vmin=0, vmax=25)

    # --- Plot delta_p predicted ---
    for idx in slices_indices:
        Z = np.full_like(X, idx)
        alpha = 0.9 if idx == 50 else 0.7  # Adjust transparency
        facecolors = get_facecolors(masked_deltap[idx, :, :], norm, cmap)
        ax_deltap.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=1, alpha=alpha, linewidth=0, edgecolor='none')

    ax_deltap.set_title("delta_p predicted", fontsize=14, fontweight='bold', pad=5)

    # --- Plot CFD results ---
    for idx in slices_indices:
        Z = np.full_like(X, idx)
        alpha = 0.9 if idx == 50 else 0.7
        facecolors = get_facecolors(masked_cfd[idx, :, :], norm, cmap)
        ax_cfd.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=1, alpha=alpha, linewidth=0, edgecolor='none')

    ax_cfd.set_title("CFD results", fontsize=14, fontweight='bold', pad=5)

    # --- Plot error field ---
    for idx in slices_indices:
        Z = np.full_like(X, idx)
        alpha = 0.9 if idx == 50 else 0.7
        facecolors = get_facecolors(masked_error[idx, :, :], error_norm, cmap)
        ax_error.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=1, alpha=alpha, linewidth=0, edgecolor='none')

    ax_error.set_title("Error (%)", fontsize=14, fontweight='bold', pad=5)

    # Adjust plot appearance
    for ax in [ax_deltap, ax_cfd, ax_error]:
        ax.set_box_aspect([4, 1, 1])
        ax.view_init(elev=30, azim=-90)
        ax.grid(False)
        ax.set_axis_off()

    # Add colorbar for error field
    mappable_error = cm.ScalarMappable(cmap=cmap, norm=error_norm)
    mappable_error.set_array(masked_error)
    cbar = fig.colorbar(mappable_error, ax=ax_error, shrink=0.6, orientation='vertical')
    cbar.set_label("Error (%)", fontsize=12)

    # Reduce extra spacing
    plt.subplots_adjust(top=0.9)  # Adjust title spacing
    plt.tight_layout()

    # Show or save the plot
    if fig_path:
        plt.savefig(fig_path)
        plt.close(fig)
    else:
        plt.show()

def plot_inputs_slices(ux, uy, uz, sdf, deltap, slices_indices=[5, 50, 95], fig_path=None):
    """
    Plot slices of the input fields: ux, uy, uz, and density.

    Args:
        ux (ndarray): Velocity field in the x-direction.
        uy (ndarray): Velocity field in the y-direction.
        uz (ndarray): Velocity field in the z-direction.
        density (ndarray): Density field.
        slices_indices (list): Indices of slices to plot.
        fig_path (str, optional): Path to save the figure. If None, the plot is displayed.
    """
    # Create the figure and axes
    fig, axs = plt.subplots(len(slices_indices), 5, figsize=(20, 5 * len(slices_indices)))

    # Set the column titles
    axs[0, 0].set_title("ux", fontsize=14, fontweight='bold')
    axs[0, 1].set_title("uy", fontsize=14, fontweight='bold')
    axs[0, 2].set_title("uz", fontsize=14, fontweight='bold')
    axs[0, 3].set_title("SDF", fontsize=14, fontweight='bold')
    axs[0, 4].set_title("output - deltaP", fontsize=14, fontweight='bold')

    # Plot the slices for each field
    for i, idx in enumerate(slices_indices):
        axs[i, 0].imshow(ux[idx, :, :], cmap='viridis', origin='lower')
        axs[i, 0].set_title(f"Slice {idx}/100", fontsize=12, fontweight='bold', loc='left')
        axs[i, 0].axis("off")

        axs[i, 1].imshow(uy[idx, :, :], cmap='viridis', origin='lower')
        axs[i, 1].set_title(f"Slice {idx}/100", fontsize=12, fontweight='bold', loc='left')
        axs[i, 1].axis("off")

        axs[i, 2].imshow(uz[idx, :, :], cmap='viridis', origin='lower')
        axs[i, 2].set_title(f"Slice {idx}/100", fontsize=12, fontweight='bold', loc='left')
        axs[i, 2].axis("off")

        axs[i, 3].imshow(sdf[idx, :, :], cmap='viridis', origin='lower')
        axs[i, 3].set_title(f"Slice {idx}/100", fontsize=12, fontweight='bold', loc='left')
        axs[i, 3].axis("off")

        axs[i, 4].imshow(deltap[idx, :, :], cmap='viridis', origin='lower')
        axs[i, 4].set_title(f"Slice {idx}/100", fontsize=12, fontweight='bold', loc='left')
        axs[i, 4].axis("off")

    # Adjust layout for better spacing
    plt.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between rows
    plt.tight_layout()

    # Show or save the plot
    if fig_path:
        plt.savefig(fig_path)
        plt.close(fig)
    else:
        plt.show()

        
def plot_delta_p_comparison_slices(cfd_results, field_deltap, no_flow_bool, slices_indices=[5, 50, 95], fig_path=None):
    """
    Plot comparison of delta_p predicted, CFD results, and error using 2D slices.

    Args:
        cfd_results (ndarray): CFD results array.
        field_deltap (ndarray): Predicted delta_p field.
        no_flow_bool (ndarray): Boolean mask for no-flow regions.
        slices_indices (list): Indices of slices to plot.
        fig_path (str, optional): Path to save the figure. If None, the plot is displayed.
    """
    # Mask the error field
    error = np.abs((field_deltap - cfd_results) / (np.max(cfd_results) - np.min(cfd_results)) * 100)
    masked_error = np.where(no_flow_bool, np.nan, error)

    # Masking the predicted delta_p field
    masked_deltap = np.where(no_flow_bool, np.nan, field_deltap)

    # Mask the CFD results
    masked_cfd = np.where(no_flow_bool, np.nan, cfd_results)

    # Set colormap and normalization
    cmap = cm.viridis
    vmin = min(np.nanmin(masked_cfd), np.nanmin(masked_deltap))
    vmax = max(np.nanmax(masked_cfd), np.nanmax(masked_deltap))
    norm = Normalize(vmin=vmin, vmax=vmax)
    error_norm = Normalize(vmin=0, vmax=25)

    # Create figure and axes
    fig, axs = plt.subplots(len(slices_indices), 3, figsize=(15, 5 * len(slices_indices)))

    # Plot slices
    for i, idx in enumerate(slices_indices):
        # Plot delta_p predicted
        axs[i, 0].imshow(masked_deltap[idx, :, :], cmap=cmap, norm=norm, origin='lower')
        axs[i, 0].set_title(f"delta_p predicted (Slice {idx})", fontsize=12, fontweight='bold')
        axs[i, 0].axis("off")

        # Plot CFD results
        axs[i, 1].imshow(masked_cfd[idx, :, :], cmap=cmap, norm=norm, origin='lower')
        axs[i, 1].set_title(f"CFD results (Slice {idx})", fontsize=12, fontweight='bold')
        axs[i, 1].axis("off")

        # Plot error field
        axs[i, 2].imshow(masked_error[idx, :, :], cmap=cmap, norm=error_norm, origin='lower')
        axs[i, 2].set_title(f"Error (%) (Slice {idx})", fontsize=12, fontweight='bold')
        axs[i, 2].axis("off")

    # Adjust layout
    plt.tight_layout()

    # Show or save the plot
    if fig_path:
        plt.savefig(fig_path)
        plt.close(fig)
    else:
        plt.show()

def define_sample_indexes(
  n_samples_per_frame,
  block_size,
  grid_res,
  first_sim,
  last_sim,
  first_t,
  last_t,
  dataset_path,
  output_pkl_path=None
):
  # Read domain bounds from the dataset
  with h5py.File(dataset_path, "r") as f:
    data = np.array(f["sim_data"], dtype='float32')
  indice = index(data[0, 0, :, 0], -100.0)[0]
  data_limited = data[0, 0, :indice, :]

  x_min = round(np.min(data_limited[..., 4]), 2)
  x_max = round(np.max(data_limited[..., 4]), 2)
  y_min = round(np.min(data_limited[..., 5]), 2)
  y_max = round(np.max(data_limited[..., 5]), 2)
  z_min = round(np.min(data_limited[..., 6]), 2)
  z_max = round(np.max(data_limited[..., 6]), 2)

  indices_per_sim_per_time = []
  for i_sim in range(first_sim, last_sim):
    indices_per_time = []
    for i_time in range(last_t - first_t):
      lower_bound = np.array([
        0 + block_size * grid_res / 2,
        0 + block_size * grid_res / 2,
        0 + block_size * grid_res / 2
      ])
      upper_bound = np.array([
        (z_max - z_min) - block_size * grid_res / 2,
        (y_max - y_min) - block_size * grid_res / 2,
        (x_max - x_min) - block_size * grid_res / 2
      ])
      ZYX = lower_bound + (upper_bound - lower_bound) * lhs(3, n_samples_per_frame)
      ZYX_indices = (np.round(ZYX / grid_res)).astype(int)
      ZYX_indices = np.unique([tuple(row) for row in ZYX_indices], axis=0)
      indices_per_time.append(ZYX_indices)
    indices_per_sim_per_time.append(indices_per_time)

  # Save to file if output_pkl_path is provided
  if output_pkl_path is not None:
    with open(output_pkl_path, 'wb') as f:
      pk.dump(indices_per_sim_per_time, f)

  return indices_per_sim_per_time

def sample_blocks(
  block_size,
  first_sim,
  last_t,
  self_first_sim,
  self_last_t,
  sim,
  t_start,
  t_end,
  calculate_maxs=False,
  sample_indices=None,
  gridded_h5_fn=None,
  max_abs_delta_Ux=0,
  max_abs_delta_Uy=0,
  max_abs_delta_Uz=0,
  max_abs_dist=0,
  max_abs_delta_p=0
):
  """Sample N blocks from each time step based on LHS"""

  inputs_u_list = []
  inputs_obst_list = []
  outputs_list = []

  count = 0
  sim_idx = sim - self_first_sim

  for time in range(t_start, t_end):

    with tables.open_file(gridded_h5_fn, mode='r') as f:
      grid = f.root.data[sim_idx * (self_last_t - self_first_sim) + time, :, :, :, :]

    ZYX_indices = sample_indices[sim_idx][time]

    for [ii, jj, kk] in ZYX_indices:

      i_idx_first = int(ii - block_size / 2)
      i_idx_last = int(ii + block_size / 2)

      j_idx_first = int(jj - block_size / 2)
      j_idx_last = int(jj + block_size / 2)

      k_idx_first = int(kk - block_size / 2)
      k_idx_last = int(kk + block_size / 2)

      inputs_u_sample = grid[i_idx_first:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, 0:3]
      inputs_obst_sample = grid[i_idx_first:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, 3:4]
      outputs_sample = grid[i_idx_first:i_idx_last, j_idx_first:j_idx_last, k_idx_first:k_idx_last, 4:5]

      # Remove all the blocks with delta_U = 0 and delta_p = 0
      if not ((inputs_u_sample == 0).all() and (outputs_sample == 0).all()):
        inputs_u_list.append(inputs_u_sample)
        inputs_obst_list.append(inputs_obst_sample)
        outputs_list.append(outputs_sample)
      else:
        count += 1

  inputs_u = np.array(inputs_u_list)
  inputs_obst = np.array(inputs_obst_list)
  outputs = np.array(outputs_list)

  # Remove mean from each output block
  for step in range(outputs.shape[0]):
    outputs[step, ...][inputs_obst[step, ...] != 0] -= np.mean(outputs[step, ...][inputs_obst[step, ...] != 0])

  print('Removing duplicate blocks ...', flush=True)
  array = np.c_[inputs_u, inputs_obst, outputs]
  reshaped_array = array.reshape(array.shape[0], -1)
  # Find unique rows
  unique_indices = np.unique(reshaped_array, axis=0, return_index=True)[1]
  unique_array = array[unique_indices]
  inputs_u, inputs_obst, outputs = unique_array[..., 0:3], unique_array[..., 3:4], unique_array[..., 4:5]

  if calculate_maxs:
    max_abs_delta_Ux = max(np.abs(inputs_u[..., 0]).max(), max_abs_delta_Ux)
    max_abs_delta_Uy = max(np.abs(inputs_u[..., 1]).max(), max_abs_delta_Uy)
    max_abs_delta_Uz = max(np.abs(inputs_u[..., 2]).max(), max_abs_delta_Uz)
    max_abs_dist = max(np.abs(inputs_obst).max(), max_abs_dist)
    max_abs_delta_p = max(np.abs(outputs).max(), max_abs_delta_p)

  if count > 0:
    print(f'    {count} blocks discarded')

  return inputs_u, inputs_obst, outputs, max_abs_delta_Ux, max_abs_delta_Uy, max_abs_delta_Uz, max_abs_dist, max_abs_delta_p


def calculate_and_save_block_abs_max(
  first_sim,
  last_sim,
  last_t,
  sample_indices_fn,
  gridded_h5_fn,
  block_size
):
  max_abs_delta_Ux = 0
  max_abs_delta_Uy = 0
  max_abs_delta_Uz = 0
  max_abs_dist = 0
  max_abs_delta_p = 0

  with open(sample_indices_fn, 'rb') as f:
    sample_indices_per_sim_per_time = pk.load(f)

  print('Calculating absolute maxs to normalize data...')
  for sim in range(first_sim, last_sim):
    for time in range(last_t - first_sim):
      _, _, _, max_abs_delta_Ux, max_abs_delta_Uy, max_abs_delta_Uz, max_abs_dist, max_abs_delta_p = sample_blocks(
        block_size,
        first_sim,
        last_t,
        first_sim,
        last_t,
        sim,
        t_start=time,
        t_end=time + 1,
        calculate_maxs=True,
        sample_indices=sample_indices_per_sim_per_time,
        gridded_h5_fn=gridded_h5_fn,
        max_abs_delta_Ux=max_abs_delta_Ux,
        max_abs_delta_Uy=max_abs_delta_Uy,
        max_abs_delta_Uz=max_abs_delta_Uz,
        max_abs_dist=max_abs_dist,
        max_abs_delta_p=max_abs_delta_p
      )

  np.savetxt('maxs', [
    max_abs_delta_Ux,
    max_abs_delta_Uy,
    max_abs_delta_Uz,
    max_abs_dist,
    max_abs_delta_p
  ])


#### ASSEMBLE ALGORITHM

def correct_pred(field_block, bool_block, i, j, k, p_i, p_j, p_k, shape, overlap, n_x, n_z, BC_col, BC_rows, BC_depths, Ref_BC):
    """
    Standalone version of _correct_pred for block correction.

    Args:
        field_block (ndarray): Block of field values.
        bool_block (ndarray): Boolean mask for the block.
        i, j, k (int): Block indices (depth, row, column).
        p_i, p_j, p_k (int): Index offsets for block placement.
        shape (int): Block shape.
        overlap (int): Overlap size.
        n_x (int): Number of blocks in x-direction.
        n_z (int): Number of blocks in z-direction.
        BC_col, BC_rows, BC_depths: Boundary condition arrays (can be None for stateless use).
        Ref_BC: Reference boundary condition (can be None for stateless use).

    Returns:
        ndarray: Corrected field block.
    """

    # i - depth index
    # j - row index
    # k - column index

    intersect_zone_limit_i = (-p_i-overlap, -p_i)
    intersect_zone_limit_j = (-p_j-overlap, -p_j)
    intersect_zone_limit_k = overlap - p_k

    # left_most_k = len(BC_rows) - 1
    down_most_j = BC_depths.shape[0] - 1

    # Case 1 - 1st correction - based on the outlet fixed pressure boundary condition (Ref_BC)
    if (i, j, k) == (0, 0, n_x-1):
        # check value at outlet BC
        if ~(bool_block[:, :, -1] == 0).all():
            BC_corr = np.mean(field_block[:, :, -1][bool_block[:, :, -1] != 0]) - Ref_BC
        else:
            BC_corr = np.mean(field_block[:, :, -2][bool_block[:, :, -2] != 0]) - Ref_BC

        field_block -= BC_corr
        BC_col = np.mean(field_block[:, :, :overlap][bool_block[:, :, :overlap] != 0])
        BC_rows[k] = np.mean(field_block[:, -overlap:, :][bool_block[:, -overlap:, :] != 0])
        BC_depths[j, k] = np.mean(field_block[-overlap:, :, :][bool_block[-overlap:, :, :] != 0])

    # Case 2 - 1st depth and 1st row - correct from the left
    elif (i, j) == (0, 0):
        # Case 2 a)
        if k > 0:
            BC_corr = np.mean(field_block[:, :, -overlap:][bool_block[:, :, -overlap:] != 0]) - BC_col
            field_block -= BC_corr
            # left-most column
            if k == 0:
                BC_col = np.mean(field_block[:, :, :intersect_zone_limit_k][bool_block[:, :, :intersect_zone_limit_k] != 0])
            else:
                BC_col = np.mean(field_block[:, :, :overlap][bool_block[:, :, :overlap] != 0])
        # Case 2 b) - Left-most block
        else:
            BC_corr = np.mean(field_block[:, :, -intersect_zone_limit_k:][bool_block[:, :, -intersect_zone_limit_k:] != 0]) - BC_col
            field_block -= BC_corr

        BC_rows[k] = np.mean(field_block[:, -overlap:, :][bool_block[:, -overlap:, :] != 0])
        BC_depths[j, k] = np.mean(field_block[-overlap:, :, :][bool_block[-overlap:, :, :] != 0])

    # Case 3 - 1st depth (non 1st row and column)
    # Correction based on the
    elif i == 0:
        # Case 3 a)
        if j < down_most_j:
            BC_corr = np.mean(field_block[:, :overlap, :][bool_block[:, :overlap, :] != 0]) - BC_rows[k]
            field_block -= BC_corr

            # Value stored to be used in the last row depends on p_i
            if j == down_most_j - 1:
                BC_rows[k] = np.mean(field_block[:, -(shape-p_j):, :][bool_block[:, -(shape-p_j):, :] != 0])
            else:
                BC_rows[k] = np.mean(field_block[:, -overlap:, :][bool_block[:, -overlap:, :] != 0])
        # Case 3 b) - Last Row
        else:
            # j_0 = #intersect_zone_limit_j[0]
            # j_f = #intersect_zone_limit_j[1]
            BC_corr = np.mean(field_block[:, :-p_j, :][bool_block[:, :-p_j, :] != 0]) - BC_rows[k]
            field_block -= BC_corr

        BC_depths[j, k] = np.mean(field_block[-overlap:, :, :][bool_block[-overlap:, :, :] != 0])

    # Case 4 - non 1st depth (any row and column)
    # Correcting based on depth overlap -> correcting (i, j, k) = (i, 0, 0) for the pressure BC could improve respecting the outlet BC
    elif i < n_z - 1:
        BC_corr = np.mean(field_block[:overlap, :, :][bool_block[:overlap, :, :] != 0]) - BC_depths[j, k]
        field_block -= BC_corr
        BC_depths[j, k] = np.mean(field_block[-overlap:, :, :][bool_block[-overlap:, :, :] != 0])

    # Case 5 - last depth
    else:
        i_0 = intersect_zone_limit_j[0]
        i_f = intersect_zone_limit_j[1]
        BC_corr = np.mean(field_block[i_0:i_f, :, :][bool_block[i_0:i_f, :, :] != 0]) - BC_depths[j, k]
        field_block -= BC_corr

    # # DEBUGGING print
    # print(f"(i,j,k): {(i,j,k)}")
    return field_block


def assemble_prediction(
    array,
    indices_list,
    n_x,
    n_y,
    n_z,
    overlap,
    shape,
    Ref_BC,
    x_array,
    apply_filter,
    shape_x,
    shape_y,
    shape_z,
    deltaU_change_grid,
    deltaP_prev_grid,
    apply_deltaU_change_wgt,
):
    """
    Reconstructs the flow domain based on squared blocks.
    In the first row the correction is based on the outlet fixed value BC.

    In the following rows the correction is based on the overlap region at the top of each new block.
    This correction from the top ensures better agreement between different rows, leading to overall better results.

    Args:
        array (ndarray): The array containing the predicted flow fields for each block.
        indices_list (list): The list of indices representing the position of each block in the flow domain.
        n_x (int): The number of blocks in the x-direction.
        n_y (int): The number of blocks in the y-direction.
        n_z (int): The number of blocks in the z-direction.
        overlap (int): Overlap size between blocks.
        shape (int): Size of each block.
        Ref_BC: Reference boundary condition (not used directly here).
        x_array (ndarray): Array with block information.
        apply_filter (bool): Whether to apply a Gaussian filter.
        shape_x (int): Domain size in x.
        shape_y (int): Domain size in y.
        shape_z (int): Domain size in z.
        deltaU_change_grid (ndarray): Grid for deltaU change weighting.
        deltaP_prev_grid (ndarray): Previous deltaP grid.
        apply_deltaU_change_wgt (bool): Whether to apply deltaU change weighting.

    Returns:
        tuple: (reconstructed domain, change_in_deltap if requested)
    """

    result_array = np.empty(shape=(shape_z, shape_y, shape_x))

    # Arrays to store average pressure in overlap regions
    # BC_col - correction between side by side blocks
    # BC_rows - correction between top and down blocks
    # BC_depth - correction between blocks in the depth direction
    BC_col = 0.0
    BC_rows = np.zeros(n_x)
    BC_depths = np.zeros((n_y, n_x))

    print(f'Shape: {(shape_z, shape_y, shape_x)}')

    # i index where the lower blocks are located
    p_i = shape_z - ((shape - overlap) * (n_z - 2) + shape)
    # j index where the left-most blocks are located
    p_j = shape_y - ((shape - overlap) * (n_y - 2) + shape)
    # k index where the left-most blocks are located
    p_k = shape_x - ((shape - overlap) * (n_x - 1) + shape)

    result = result_array

    # Loop over all the blocks and apply corrections to ensure consistency between overlapping blocks
    for i_block in range(x_array.shape[0]):
        i, j, k = indices_list[i_block]
        flow_bool = x_array[i_block, :, :, :, 3]
        pred_field = array[i_block, ...]

        # Applying the correction
        pred_field = correct_pred(
            pred_field, flow_bool, i, j, k, p_i, p_j, p_k,
            shape=shape, overlap=overlap, n_x=n_x, n_z=n_z,
            BC_col=BC_col, BC_rows=BC_rows, BC_depths=BC_depths, Ref_BC=Ref_BC
        )

        # Last reassembly step:
        # Assigning the block to the right location in the flow domain

        # # DEBUGGING print
        # print(f"(i,j,k): {(i,j,k)}")

        # Non last depth
        if i < n_z - 1:
            # Last row, first column (right-most)
            if (j, k) == (n_y - 1, 0):
                # # DEBUGGING print
                # print((shape - overlap) * i, (shape - overlap) * i + shape, shape_y - p_j, shape_y, 0, shape)
                result[(shape - overlap) * i:(shape - overlap) * i + shape,
                       -p_j:shape_y, :shape] = pred_field[:, -p_j:, :]
            # Last column (left-most)
            elif k == 0:
                # # DEBUGGING print
                # print((shape - overlap) * i, (shape - overlap) * i + shape,
                #       (shape - overlap) * j, (shape - overlap) * j + shape,
                #       0, shape)
                result[(shape - overlap) * i:(shape - overlap) * i + shape,
                       (shape - overlap) * j:(shape - overlap) * j + shape,
                       0:shape] = pred_field
            # Last row
            elif j == (n_y - 1):
                k_ = n_x - k - 1
                # # DEBUGGING print
                # print((shape - overlap) * i, (shape - overlap) * i + shape,
                #       shape_y - p_j, shape_y, shape_x - shape - k_ * (shape - overlap), shape_x - k_ * (shape - overlap))
                result[(shape - overlap) * i:(shape - overlap) * i + shape,
                       -p_j:,
                       shape_x - shape - k_ * (shape - overlap): shape_x - k_ * (shape - overlap)] = pred_field[:, -p_j:, :]
            else:
                k_ = n_x - k - 1
                # # DEBUGGING print
                # print((shape - overlap) * i, (shape - overlap) * i + shape,
                #       (shape - overlap) * j, (shape - overlap) * j + shape,
                #       shape_x - shape - k_ * (shape - overlap), shape_x - k_ * (shape - overlap))
                result[(shape - overlap) * i:(shape - overlap) * i + shape,
                       (shape - overlap) * j:(shape - overlap) * j + shape,
                       shape_x - shape - k_ * (shape - overlap): shape_x - k_ * (shape - overlap)] = pred_field
        # Last depth
        else:
            # Last row, first column (right-most)
            if (j, k) == (n_y - 1, 0):
                # # DEBUGGING print
                # print((shape_z - p_i, shape_z), (shape_y - p_j, shape_y), 0, shape)
                result[-p_i:,
                       -p_j:, :shape] = pred_field[-p_i:, -p_j:, :]
            # Last column (left-most)
            elif k == 0:
                # # DEBUGGING print
                # print(shape_z - p_i, shape_z, shape_y - p_j, shape_y, 0, shape)
                result[-p_i:,
                       (shape - overlap) * j:(shape - overlap) * j + shape,
                       0:shape] = pred_field[-p_i:, :, :]
            # Last row
            elif j == (n_y - 1):
                k_ = n_x - k - 1
                # # DEBUGGING print
                # print(shape_z - p_i, shape_z, shape_y - p_j, shape_y, shape_x - shape - k_ * (shape - overlap), shape_x - k_ * (shape - overlap))
                result[-p_i:,
                       -p_j:,
                       shape_x - shape - k_ * (shape - overlap): shape_x - k_ * (shape - overlap)] = pred_field[-p_i:, -p_j:, :]
            else:
                k_ = n_x - k - 1
                # # DEBUGGING print
                # print(shape_z - p_i, shape_z, (shape - overlap) * j, (shape - overlap) * j + shape, shape_x - shape - k_ * (shape - overlap), shape_x - k_ * (shape - overlap))
                result[-p_i:,
                       (shape - overlap) * j:(shape - overlap) * j + shape,
                       shape_x - shape - k_ * (shape - overlap): shape_x - k_ * (shape - overlap)] = pred_field[-p_i:, :, :]

        # # DEBUGGING: plot slices
        # if i==0: # and j==3:
        #     fig, axs = plt.subplots(7, 1, figsize=(15, 5))
        #     axs[0].imshow(result[(shape-overlap) * i + 1, :, :])
        #     axs[1].imshow(result[(shape-overlap) * i + 3, :,:])
        #     axs[2].imshow(result[(shape-overlap) * i + 5, :,:])
        #     axs[3].imshow(result[(shape-overlap) * i + 7, :,:])
        #     axs[4].imshow(result[(shape-overlap) * i + 9,:,:])
        #     axs[5].imshow(result[(shape-overlap) * i + 11,:,:])
        #     axs[6].imshow(result[(shape-overlap) * i + 13,:,:])
        #     for ax in axs:
        #         plt.colorbar(ax.images[0], ax=ax)
        #     plt.savefig(f"reconstruct/reconstructed_{i_block}.png")
        #     plt.close(fig)

    # Correction based on the fact the BC is applied at the last cell center and not the cell face...
    if ~(flow_bool[:, :, -1] == 0).all():
        result -= np.mean(3 * result[:, :, -1] - result[:, :, -2]) / 3
    else:
        result -= np.mean(3 * result[:, :, -2] - result[:, :, -3]) / 3

    ################### this applies a gaussian filter to remove boundary artifacts #################
    filter_tuple = (10, 10, 10)
    if apply_filter:
        result = ndimage.gaussian_filter(result, sigma=filter_tuple, order=0)

    change_in_deltap = None
    if apply_deltaU_change_wgt:
        deltaU_change_grid = ndimage.gaussian_filter(deltaU_change_grid, sigma=(5, 5, 5), order=0)
        change_in_deltap = result - deltaP_prev_grid
        change_in_deltap = change_in_deltap * deltaU_change_grid
        change_in_deltap = ndimage.gaussian_filter(change_in_deltap, sigma=filter_tuple, order=0)

    return result, change_in_deltap