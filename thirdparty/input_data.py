# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This file has been modified from the original code released by Google Inc.
# in the Tensorflow library. The Apache License for this file
# can be found in the directory thirdparty.
# The modifications in this file are subject to the same Apache License, Version 2.0.
#
# Copyright 2018 D-Wave Systems Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading Binarized MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import numpy as np
import os
import scipy.io

from util import binarize

import numpy
from six.moves import urllib


# This function is added to load numpy files.
def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    data = np.load(filename)
    data = np.asarray(data['arr_0'], dtype=np.uint8)
    data = data.reshape(data.shape[0], 28, 28, 1)
    return data


# This function is created from the original extract_images function to load binary images.
def extract_data_continuous(filename, num_images):
    """Extract the images into a 2D tensor [image index, pixels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    IMAGE_SIZE = 784
    PIXEL_DEPTH = 255

    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        # data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data /= PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE)
        return data


# This function is created based on the original maybe_download function to download
# a list of mnist files and also convert them to numpy version.
# Instead of the original MNIST, the binarized MNIST is downloaded.
def maybe_download_binarized_mnist(WORK_DIRECTORY):
    """Download the data, unless it's already here."""
    SOURCE_URL = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/'
    if not os.path.exists(WORK_DIRECTORY):
        os.mkdir(WORK_DIRECTORY)

    filenames = ['binarized_mnist_train', 'binarized_mnist_valid', 'binarized_mnist_test']
    for filename in filenames:
        filepath = os.path.join(WORK_DIRECTORY, filename + '.npz')
        if not os.path.exists(filepath):
            temp_filepath = os.path.join(WORK_DIRECTORY, filename + '.amat')
            if not os.path.exists(temp_filepath):
                temp_filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename + '.amat', temp_filepath)
                statinfo = os.stat(temp_filepath)
                print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

            data = np.loadtxt(temp_filepath)
            np.savez_compressed(filepath, data)
            os.remove(temp_filepath)
            print('Successfully converted')
    return


# This is a new function introduced for downloading OMNIGLOT dataset.
def maybe_download_omniglot(WORK_DIRECTORY):
    """Download the data, unless it's already here."""
    SOURCE_URL = 'https://raw.github.com/yburda/iwae/master/datasets/OMNIGLOT/'
    if not os.path.exists(WORK_DIRECTORY):
        os.mkdir(WORK_DIRECTORY)

    filename = 'chardata.mat'
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    return


# In the following class, fake data mechanism has been removed.
class DataSet(object):
    def __init__(self, images, should_binarize=False, one_hot=False):
        """Construct a DataSet. one_hot arg is used only if fake_data is true."""
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        # assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2] * images.shape[3])
        self._images = images
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.should_binarize = should_binarize

    @property
    def images(self):
        return self._images

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        batch = self._images[start:end]
        return batch.astype(numpy.float32), None


class DataSets(object):
    pass


# This function has been modified based on the original read_data_sets function.
# Instead of the original MNIST dataset, we load the numpy files created above.
def read_binarize_mnist(train_dir):
    data_sets = DataSets()

    TRAIN_IMAGES = os.path.join(train_dir, 'binarized_mnist_train.npz')
    VALIDATION_IMAGES = os.path.join(train_dir, 'binarized_mnist_valid.npz')
    TEST_IMAGES = os.path.join(train_dir, 'binarized_mnist_test.npz')

    train_images = extract_images(TRAIN_IMAGES)
    validation_images = extract_images(VALIDATION_IMAGES)
    test_images = extract_images(TEST_IMAGES)

    data_sets.train = DataSet(train_images, should_binarize=False)
    data_sets.validation = DataSet(validation_images, should_binarize=False)
    data_sets.test = DataSet(test_images, should_binarize=False)

    return data_sets


# A new function introduced for load the OMNIGLOT dataset.
def read_omniglot(data_dir, binarize_val_test=False):
    data_path = os.path.join(data_dir, 'chardata.mat')

    def reshape_data(unshaped_data):  # matlab is column major; convert to row-major as in numpy
        return unshaped_data.reshape((-1, 28, 28)).reshape((-1, 28 * 28), order='fortran')

    omni_raw = scipy.io.loadmat(data_path)

    num_validation = 1345
    raw_train_data = reshape_data(omni_raw['data'].T.astype('float32'))

    # form splits
    train_data = raw_train_data[:-num_validation, :]
    validation_data = raw_train_data[-num_validation:, :]
    test_data = reshape_data(omni_raw['testdata'].T.astype('float32'))

    # NOTE: we already binarize the validation and test using a fixed seed.
    # For the train dataset we set the should_binarize
    # to True so that we dynamically binarize it as we are loading the batches.
    if binarize_val_test:
        validation_data = binarize(validation_data, seed=1)
        test_data = binarize(test_data, seed=1)

    data_sets = DataSets()
    data_sets.train = DataSet(train_data.reshape((-1, 28, 28, 1)), should_binarize=binarize_val_test)
    data_sets.validation = DataSet(validation_data.reshape((-1, 28, 28, 1)), should_binarize=False)
    data_sets.test = DataSet(test_data.reshape((-1, 28, 28, 1)), should_binarize=False)

    return data_sets


# A new function introduced for loading datasets.
def read_data_set(data_dir, dataset):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if dataset == 'binarized_mnist':
        data_dir = os.path.join(data_dir, 'binarized_mnist')
        maybe_download_binarized_mnist(data_dir)
        return read_binarize_mnist(data_dir)
    elif dataset == 'omniglot':
        data_dir = os.path.join(data_dir, 'OMNIGLOT')
        maybe_download_omniglot(data_dir)
        return read_omniglot(data_dir, binarize_val_test=True)
    else:
        raise NotImplementedError
