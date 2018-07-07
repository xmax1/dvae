# Copyright 2018 D-Wave Systems Inc.
# DVAE# licensed to authorized users only under the applicable license
# agreement.  See LICENSE.

import tensorflow as tf
import numpy as np
import itertools

from time import localtime, strftime


def Print(st):
    """ This function can be used to print logs. In our cluster, it will properly write to the cluster log. 
    
    Args:
        st: is a string.

    Returns:
        None
    """
    assert isinstance(st, str), 'st should be a string.'
    time_st = strftime("[%Y %b %d %H:%M:%S] ", localtime())
    print(time_st + st)


def binarize(data, seed=None):
    """ This function binarizes the input numpy array assuming that values are representing the mean parameter in a
    Bernoulli distribution.
    
    Args:
        data: a numpy array with values \in [0, 1]
        seed: seed used for numpy.
    Returns:
        bin: binarized sample.
    """
    if seed is not None:
        np.random.seed(seed)

    random = np.random.rand(*data.shape[:])
    bin = np.asarray(random < data, np.float32)
    return bin


def repeat_input_iw(input, k):
    """ This function repeats the input tensor in the column dimensions.

    Args:
        input: a tensor in the shape batch_size x num_columns
        k: number of repetitions.
    Returns:
        repeated: repeated tensor.
    """
    repeated = tf.reshape(tf.tile(input, [1, k]), [-1, input.get_shape().as_list()[1]])
    return repeated


def fill_feed_dict(data_set, images_pl, batch_size):
    """Fills the feed_dict for training the given step.

    Args:
        data_set: The set of images, from input_data.read_data_sets()
        images_pl: The images placeholder.
        batch_size: batch size

    Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size ` examples.
    images_feed, _ = data_set.next_batch(batch_size)

    if data_set.should_binarize:
        images_feed = binarize(images_feed)

    feed_dict = {images_pl: images_feed}
    return feed_dict


def create_reconstruction_image(input, reconstruction, batch_size):
    """ Given the current batch and the reconstructed p(x|z), this function creates an image tensor containing
     tiled input and reconstructed instances.

    Args:
        input: a tensor containing vectorized instances (x) in each row.
        reconstruction: a tensor containing vectorized reconstructed instances using p(x|z) in each row.
        batch_size: the number of rows in input or reconstruction tensors.

    Returns:
        image: containing tiled input and reconstruction images.
    """
    assert input.get_shape().ndims == 2, 'input should be 2D vectorized images.'
    assert reconstruction.get_shape().ndims == 2, 'reconstruction should be 2D vectorized images.'
    num_input = input.get_shape().as_list()[1]
    image_size = int(np.sqrt(num_input))
    shape = [batch_size, image_size, image_size]
    original = tf.cast(255 * tf.reshape(input, shape), tf.uint8)
    recon = tf.cast(255 * tf.reshape(reconstruction, shape), tf.uint8)
    n = int(np.floor(np.sqrt(batch_size)))
    original = tf.slice(original, [0, 0, 0], [n*n, -1, -1])
    recon = tf.slice(recon, [0, 0, 0], [n * n, -1, -1])
    original = tile_image_tf(original, n, n, image_size, image_size)
    recon = tile_image_tf(recon, n, n, image_size, image_size)
    image = tf.concat(axis=1, values=[original, recon])
    return image


def tile_image_tf(images, n, m, height, width):
    """Tile images from a 3D input tensor. This function creates a large image by tiling n images vertically 
    and m images horizontally. 

    Args:
        images: A tensor of size [n*m x image_height*image_width] or [n*m x image_height x image_width] 
        n: number of images tiled vertically.
        m: number of images tiled horizontally.
        height: image height.
        width: image width.

    Returns:
        tiled_image: A 4D tensor of shape 1 x n * height x m * width x 1 created by tiling images in rows and columns.
    """
    assert images.get_shape().ndims == 3 or images.get_shape().ndims == 2, 'image should be 2D or 3D.'
    shape = images.get_shape().as_list()
    assert shape[1] == height * width or (shape[1] == height and shape[2] == width), \
        'image dims should match height and width.'
    tiled_image = tf.reshape(images, [n, m, height, width])
    tiled_image = tf.transpose(tiled_image, [0, 1, 3, 2])
    tiled_image = tf.reshape(tiled_image, [n, m * width, height])
    tiled_image = tf.transpose(tiled_image, [0, 2, 1])
    tiled_image = tf.reshape(tiled_image, [1, n * height, m * width, 1])

    return tiled_image


def get_global_step_var():
    """ returns the global_step variable. If it doesn't exist, it creates one. 
    
    Returns:
        var: a tensorflow variable, not trainable, that can be used as global step.
    """
    name = 'global_step_var'
    try:
        return tf.get_default_graph().get_tensor_by_name(name+':0')
    except KeyError:
        scope = tf.get_variable_scope()
        assert scope.name == '', "Creating global_step_var under a variable scope would cause problems!"
        var = tf.Variable(0, trainable=False, name=name)
        return var


def get_all_states(num_variables):
    """ generates all the binary vectors of size num_variables.
    
    Args:
        num_variables: the number of binary values.

    Returns:
        states: is a numpy array. Each row corresponds to a particular state. The total number of rows
        is 2^{num_variables}.
    """
    assert num_variables < 22, 'setting number of variables will cause out of memory.'
    states = np.array(list(itertools.product([0.0, 1.0], repeat=num_variables)))
    return states
