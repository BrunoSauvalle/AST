

# CLEVR (with masks) dataset reader.
# imported and adapted from https://github.com/deepmind/multi_object_datasets/blob/master/clevr_with_masks.py
# in order to use this code, you should first download the CLEVR dataset as a tfrecords from the following link provided by Deepmind :
# https://console.cloud.google.com/storage/browser/multi-object-datasets/clevr_with_masks


#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


import numpy as np

import os
import cv2
import torch
import PIL
import tqdm

import tensorflow.compat.v1 as tf


COMPRESSION_TYPE = tf.io.TFRecordOptions.get_compression_type_string('GZIP')
IMAGE_SIZE = [240, 320]
# The maximum number of foreground and background entities in the provided
# dataset. This corresponds to the number of segmentation masks returned per
# scene.
MAX_NUM_ENTITIES = 11
BYTE_FEATURES = ['mask', 'image', 'color', 'material', 'shape', 'size']

# Create a dictionary mapping feature names to `tf.Example`-compatible
# shape and data type descriptors.
features = {
    'image': tf.FixedLenFeature(IMAGE_SIZE+[3], tf.string),
    'mask': tf.FixedLenFeature([MAX_NUM_ENTITIES]+IMAGE_SIZE+[1], tf.string),
    'x': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'y': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'z': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'pixel_coords': tf.FixedLenFeature([MAX_NUM_ENTITIES, 3], tf.float32),
    'rotation': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'size': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'material': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'shape': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'color': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.string),
    'visibility': tf.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
}


def _decode(example_proto):
  # Parse the input `tf.Example` proto using the feature description dict above.
  single_example = tf.parse_single_example(example_proto, features)
  for k in BYTE_FEATURES:
    single_example[k] = tf.squeeze(tf.decode_raw(single_example[k], tf.uint8),
                                   axis=-1)
  return single_example


def dataset(tfrecords_path, read_buffer_size=None, map_parallel_calls=None):
  """Read, decompress, and parse the TFRecords file.
  Args:
    tfrecords_path: str. Path to the dataset file.
    read_buffer_size: int. Number of bytes in the read buffer. See documentation
      for `tf.data.TFRecordDataset.__init__`.
    map_parallel_calls: int. Number of elements decoded asynchronously in
      parallel. See documentation for `tf.data.Dataset.map`.
  Returns:
    An unbatched `tf.data.TFRecordDataset`.
  """
  raw_dataset = tf.data.TFRecordDataset(
      tfrecords_path, compression_type=COMPRESSION_TYPE,
      buffer_size=read_buffer_size)
  return raw_dataset.map(_decode, num_parallel_calls=map_parallel_calls)

PATH = '/workspace/nvme0n1p1/Datasets/clevr/clevr_with_masks_clevr_with_masks_train.tfrecords'

clevr_dataset = dataset(PATH)

step = 0
IMAGES_PATH = '/workspace/nvme0n1p1/Datasets/clevr'

CMAP = torch.tensor([
        [0, 0, 0],
        [255, 0, 0],
        [0, 128, 0],
        [0, 0, 255],
        [255, 255, 0],
        [141, 211, 199],
        [255, 255, 179],
        [190, 186, 218],
        [251, 128, 114],
        [128, 177, 211],
        [253, 180, 98],
        [179, 222, 105],
        [252, 205, 229],
        [217, 217, 217]])

for index, data in enumerate(tqdm(clevr_dataset)):

    image_path = os.path.join(IMAGES_PATH,'CLEVR_full_%06d.png'% index)
    image = data['image'].numpy()
    cv2.imwrite(image_path, image)

    mask_path =  os.path.join(IMAGES_PATH,'CLEVR_full_%06d_flat.png'% index)
    GT_mask = data['mask'].numpy() # 11 x 240 x 320 x 1
    scalar_mask = np.argmax(GT_mask, axis=0)
    scalar_mask = np.squeeze(scalar_mask,axis=2)
    GT_mask = PIL.Image.fromarray(scalar_mask.astype(np.uint8), mode = 'P')
    GT_mask.putpalette(CMAP.flatten().tolist() * 4)
    GT_mask.save(mask_path)



