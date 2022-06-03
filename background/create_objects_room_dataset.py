# Objects Room dataset reader.
# partially downloaded and adapted from https://github.com/deepmind/multi_object_datasets/blob/master/objects_room.py
# in order to use this code, you should first download the objectsroom dataset as a tfrecords from the following link provided by Deepmind :
# https://storage.googleapis.com/multi-object-datasets/objects_room/objects_room_train.tfrecords


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



import functools
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import os
import PIL
import numpy as np
import torch
import cv2
from tqdm import tqdm

COMPRESSION_TYPE = tf.io.TFRecordOptions.get_compression_type_string('GZIP')
IMAGE_SIZE = [64, 64]
# The maximum number of foreground and background entities in each variant
# of the provided datasets. The values correspond to the number of
# segmentation masks returned per scene.
MAX_NUM_ENTITIES = {
    'train': 7,
    'six_objects': 10,
    'empty_room': 4,
    'identical_color': 10
}
BYTE_FEATURES = ['mask', 'image']


def feature_descriptions(max_num_entities):
  """Create a dictionary describing the dataset features.
  Args:
    max_num_entities: int. The maximum number of foreground and background
      entities in each image. This corresponds to the number of segmentation
      masks returned per scene.
  Returns:
    A dictionary which maps feature names to `tf.Example`-compatible shape and
    data type descriptors.
  """
  return {
      'image': tf.FixedLenFeature(IMAGE_SIZE+[3], tf.string),
      'mask': tf.FixedLenFeature([max_num_entities]+IMAGE_SIZE+[1], tf.string),
  }


def _decode(example_proto, features):
  # Parse the input `tf.Example` proto using a feature description dictionary.
  single_example = tf.parse_single_example(example_proto, features)
  for k in BYTE_FEATURES:
    single_example[k] = tf.squeeze(tf.decode_raw(single_example[k], tf.uint8),
                                   axis=-1)
  return single_example


def dataset(tfrecords_path, dataset_variant, read_buffer_size=None,
            map_parallel_calls=None):
  """Read, decompress, and parse the TFRecords file.
  Args:
    tfrecords_path: str. Path to the dataset file.
    dataset_variant: str. One of ['train', 'six_objects', 'empty_room',
      'identical_color']. This is used to identify the maximum number of
      entities in each scene. If an incorrect identifier is passed in, the
      TFRecords file will not be read correctly.
    read_buffer_size: int. Number of bytes in the read buffer. See documentation
      for `tf.data.TFRecordDataset.__init__`.
    map_parallel_calls: int. Number of elements decoded asynchronously in
      parallel. See documentation for `tf.data.Dataset.map`.
  Returns:
    An unbatched `tf.data.TFRecordDataset`.
  """
  if dataset_variant not in MAX_NUM_ENTITIES:
    raise ValueError('Invalid `dataset_variant` provided. The supported values'
                     ' are: {}'.format(list(MAX_NUM_ENTITIES.keys())))
  max_num_entities = MAX_NUM_ENTITIES[dataset_variant]
  raw_dataset = tf.data.TFRecordDataset(
      tfrecords_path, compression_type=COMPRESSION_TYPE,
      buffer_size=read_buffer_size)
  features = feature_descriptions(max_num_entities)
  partial_decode_fn = functools.partial(_decode, features=features)
  return raw_dataset.map(partial_decode_fn,
                         num_parallel_calls=map_parallel_calls)
# The following path should be updated with the path to downladed objects_room_objects_room_train.tfrecords in your filesystem
PATH = '/workspace/nvme0n1p1/Datasets/objects_room/objects_room_objects_room_train.tfrecords'


objects_room_dataset = dataset(PATH, 'train')

step = 0
# The following path should be updated with the path where you want the objectsroom dataset to be stored in your filesystem
IMAGES_PATH = '/workspace/nvme0n1p1/Datasets/objects_room'

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


for index, data in enumerate(tqdm(objects_room_dataset)):

    image_path = os.path.join(IMAGES_PATH,'input','objects_room_input_%07d.png'% index)
    image = data['image'].numpy() # RGB
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, image)

    mask_path =  os.path.join(IMAGES_PATH,'GT_masks','objects_room_GT_mask_%07d.png'% index)
    GT_mask = data['mask'].numpy() # 11 x 240 x 320 x 1
    scalar_mask = np.maximum([0],np.argmax(GT_mask, axis=0)-3)
    scalar_mask = np.squeeze(scalar_mask,axis=2)
    GT_mask = PIL.Image.fromarray(scalar_mask.astype(np.uint8), mode = 'P')
    GT_mask.putpalette(CMAP.flatten().tolist() * 4)
    GT_mask.save(mask_path)


