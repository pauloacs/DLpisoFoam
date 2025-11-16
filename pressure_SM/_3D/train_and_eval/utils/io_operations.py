"""
Functions for reading/writing datasets, boundaries, and serialization.
"""

import os
import h5py
import numpy as np
import tensorflow as tf
import tables


def read_boundaries(sim_i, original_dataset_path):
    """
    Read boundary data for a given simulation index.
    """
    with h5py.File(original_dataset_path, "r") as f:
        obst_boundary = np.array(f["obst_bound"][sim_i, 0, ...], dtype='float32')
        y_bot_boundary = np.array(f["y_bot_bound"][sim_i, 0, ...], dtype='float32')
        z_bot_boundary = np.array(f["z_bot_bound"][sim_i, 0, ...], dtype='float32')
        y_top_boundary = np.array(f["y_top_bound"][sim_i, 0, ...], dtype='float32')
        z_top_boundary = np.array(f["z_top_bound"][sim_i, 0, ...], dtype='float32')
    
    from .data_processing import index
    
    indice_z_top = index(z_top_boundary[:,0], -100.0)[0]
    z_top_boundary = z_top_boundary[:indice_z_top, :]
    indice_y_top = index(y_top_boundary[:,0], -100.0)[0]
    y_top_boundary = y_top_boundary[:indice_y_top, :]
    indice_y_bot = index(y_bot_boundary[:,0], -100.0)[0]
    y_bot_boundary = y_bot_boundary[:indice_y_bot, :]
    indice_z_bot = index(z_bot_boundary[:,0], -100.0)[0]
    z_bot_boundary = z_bot_boundary[:indice_z_bot, :]
    indice_obst = index(obst_boundary[:,0] , -100.0 )[0]
    obst_boundary = obst_boundary[:indice_obst,:]

    boundaries = {
        'obst_boundary': obst_boundary,
        'y_bot_boundary': y_bot_boundary,
        'z_bot_boundary': z_bot_boundary,
        'y_top_boundary': y_top_boundary,
        'z_top_boundary': z_top_boundary
    }

    return boundaries


def read_cells_and_limits(original_dataset_path, sim_i, first_t, last_t):
    """
    Read cell data and limits for a given simulation index.
    """
    from .data_processing import index
    
    with h5py.File(original_dataset_path, "r") as f:
        data = np.array(f["sim_data"][sim_i:sim_i+1, first_t:(first_t + last_t), ...], dtype='float32')

    indice = index(data[0, 0, :, 0], -100.0)[0]
    data_limited = data[0, :, :indice, :]

    limits = {
        'x_min': round(np.min(data_limited[0, :, 4]), 2),
        'x_max': round(np.max(data_limited[0, :, 4]), 2),
        'y_min': round(np.min(data_limited[0, :, 5]), 2),
        'y_max': round(np.max(data_limited[0, :, 5]), 2),
        'z_min': round(np.min(data_limited[0, :, 6]), 2),
        'z_max': round(np.max(data_limited[0, :, 6]), 2)
    }

    return data_limited, limits


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
    """Define the dictionary -- the structure -- of our single example"""
    data = {
        'height': int64_feature(input_parse.shape[0]),
        'depth_x': int64_feature(input_parse.shape[1]),
        'depth_y': int64_feature(output_parse.shape[1]),
        'raw_input': bytes_feature(tf.io.serialize_tensor(input_parse)),
        'output': bytes_feature(tf.io.serialize_tensor(output_parse)),
    }

    # Create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out


def write_images_to_tfr_short(input, output, filename: str = "images"):
    """Write input/output pairs to TFRecord file."""
    filename = filename + ".tfrecords"
    # Create a writer that'll store our data to disk
    writer = tf.io.TFRecordWriter(filename)
    count = 0

    for index in range(len(input)):
        # Get the data we want to write
        current_input = input[index].astype('float32')
        current_output = output[index].astype('float32')

        out = parse_single_image(input_parse=current_input, output_parse=current_output)
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count


def parse_tfr_element(element):
    """Parse a single TFRecord element."""
    # Use the same structure as above
    data = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'output': tf.io.FixedLenFeature([], tf.string),
        'raw_input': tf.io.FixedLenFeature([], tf.string),
        'depth_x': tf.io.FixedLenFeature([], tf.int64),
        'depth_y': tf.io.FixedLenFeature([], tf.int64)
    }

    content = tf.io.parse_single_example(element, data)

    height = content['height']
    depth_x = content['depth_x']
    depth_y = content['depth_y']
    output = content['output']
    raw_input = content['raw_input']

    # Get our 'feature' -- our image -- and reshape it appropriately
    input_out = tf.io.parse_tensor(raw_input, out_type=tf.float32)
    output_out = tf.io.parse_tensor(output, out_type=tf.float32)

    return (input_out, output_out)


def load_dataset_tf(filename, batch_size, buffer_size):
    """Load dataset from TFRecord file."""
    # Create the dataset
    dataset = tf.data.TFRecordDataset(filename)

    # Pass every single feature through our mapping function
    dataset = dataset.map(parse_tfr_element)

    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)

    return dataset


def get_gridded_h5_filenames(base_gridded_h5_fn, first_sim, last_sim):
    """
    Generate a list of gridded HDF5 filenames for a range of simulations.
    
    Args:
        base_gridded_h5_fn (str): Base filename for the gridded HDF5 file (e.g., 'gridded_data.h5')
        first_sim (int): First simulation index
        last_sim (int): Last simulation index (exclusive)
    
    Returns:
        list: List of filenames for each simulation
    """
    base_name = base_gridded_h5_fn.split('.h5')[0]
    return [f"{base_name}_sim{sim_i}.h5" for sim_i in range(first_sim, last_sim + 1)]


