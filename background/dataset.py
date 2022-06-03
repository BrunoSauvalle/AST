# In order to use the CLEVRTEX dataset code, the CLEVRTEX dataset should be first downloaded
# from the following link : https://www.robots.ox.ac.uk/~vgg/data/clevrtex/
#
# Shapestacks dataset can be downloaded from the following link :
# https://drive.google.com/drive/folders/1KsSQCgb1JJExbKyrIkTwBL9VidGcq2k7
#
# Shapestacks_config code is not provided here due to copyright issue. You can download it at your own risk
# from
# https://github.com/applied-ai-lab/genesis/blob/master/datasets/shapestacks_config.py
# and then uncomment the import line


from __future__ import print_function

import os
import torch.utils.data as data
import natsort
import cv2
import json
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms.functional as Ft
from PIL import Image
#from shapestacks_config import ShapeStacksDataset
from config import env


class Image_dataset(data.Dataset):

    def __init__(self, input_dataset_path, GT_mask_dataset_path = None):

        self.input_dir = input_dataset_path

        if GT_mask_dataset_path == None:
            self.with_GT_masks = False
        else:
            self.with_GT_masks = True

        input_image_names = []
        for root, dirs, files in os.walk(self.input_dir, topdown=False):
                input_image_names+= [os.path.join(root, file) for file in files]

        input_image_names = natsort.natsorted(input_image_names)

        self.dataset_length = len(input_image_names)
        assert self.dataset_length > 1 , 'input dataset is empty'
        self.input_image_names = input_image_names

        if self.with_GT_masks:
            self.GT_mask_dir = GT_mask_dataset_path

            GT_mask_image_names = [item for item in os.listdir(self.GT_mask_dir) if
                              os.path.isfile(os.path.join(self.GT_mask_dir, item))]
            GT_mask_image_names = natsort.natsorted(GT_mask_image_names)

            self.GT_mask_image_names = GT_mask_image_names

            assert len(
                GT_mask_image_names) == self.dataset_length, f'error input dataset has length {self.dataset_length} but GT mask dataset lenght is {len(GT_mask_image_names)}'

        _, first_input_image, first_GT_mask_image = self[0]

        input_nc, self.image_height, self.image_width = first_input_image.shape
        _,GT_mask_image_height, GT_mask_image_width = first_GT_mask_image.shape

        assert GT_mask_image_height == self.image_height
        assert GT_mask_image_width == self.image_width

        print(f'dataset initialized  w = {self.image_width},h = {self.image_height} number of frames {self.dataset_length}')

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
            input_image_path = os.path.join(self.input_dir, self.input_image_names[idx])
            input_opencv_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)  # 0-255 HWC BGR
            input_opencv_image = cv2.cvtColor(input_opencv_image, cv2.COLOR_BGR2RGB)
            input_np_image = np.asarray(input_opencv_image)
            input_np_image = np.transpose(input_np_image, (2, 0, 1))  # 0-255 CHW RGB
            input_tensor_image = torch.from_numpy(input_np_image).type(torch.FloatTensor) #
            if self.with_GT_masks:
                GT_mask_image_path = os.path.join(self.GT_mask_dir, self.GT_mask_image_names[idx])
                GT_mask_PIL_image = Image.open(GT_mask_image_path )
                GT_mask_tensor_image = torch.from_numpy(np.array(GT_mask_PIL_image))[None]
            else:
                c,h,w = input_tensor_image.shape
                GT_mask_tensor_image = torch.zeros((1,h,w))
            return idx, input_tensor_image,  GT_mask_tensor_image # scalar RGB,  2D

# The following code is imported and adapted from CLEVRTEX Github  https://github.com/karazijal/clevrtex


class DatasetReadError(ValueError):
    pass

class CLEVRTEX:
    ccrop_frac = 0.8
    splits = {
        'test': (0., 0.1),
        'val': (0.1, 0.2),
        'train': (0.2, 1.)
    }
    shape = (3, 240, 320)
    variants = {'full', 'pbg', 'vbg', 'grassbg', 'camo', 'outd'}

    def _index_with_bias_and_limit(self, idx):
        if idx >= 0:
            idx += self.bias
            if idx >= self.limit:
                raise IndexError()
        else:
            idx = self.limit + idx
            if idx < self.bias:
                raise IndexError()
        return idx

    def _reindex(self):
        print(f'Indexing {self.basepath}')

        img_index = {}
        msk_index = {}
        met_index = {}
        if env.dataset_name == "clevrtex":
            print('using CLEVRTEX prefix ')
            prefix = f"CLEVRTEX_{self.dataset_variant}_"
        elif env.dataset_name == "clevr":
            prefix = f"CLEVR_{self.dataset_variant}_"

        img_suffix = ".png"
        msk_suffix = "_flat.png"
        met_suffix = ".json"

        _max = 0
        for img_path in self.basepath.glob(f'**/{prefix}??????{img_suffix}'):
            indstr = img_path.name.replace(prefix, '').replace(img_suffix, '')
            msk_path = img_path.parent / f"{prefix}{indstr}{msk_suffix}"
            met_path = img_path.parent / f"{prefix}{indstr}{met_suffix}"
            indstr_stripped = indstr.lstrip('0')
            if indstr_stripped:
                ind = int(indstr)
            else:
                ind = 0
            if ind > _max:
                _max = ind

            if not msk_path.exists():
                raise DatasetReadError(f"Missing {msk_suffix.name}")

            if ind in img_index:
                raise DatasetReadError(f"Duplica {ind}")

            img_index[ind] = img_path
            msk_index[ind] = msk_path
            if self.return_metadata:
                if not met_path.exists():
                    raise DatasetReadError(f"Missing {met_path.name}")
                met_index[ind] = met_path
            else:
                met_index[ind] = None

        if len(img_index) == 0:
            raise DatasetReadError(f"No values found")
        missing = [i for i in range(0, _max) if i not in img_index]
        if missing:
            raise DatasetReadError(f"Missing images numbers {missing}")

        return img_index, msk_index, met_index

    def _variant_subfolder(self):
        if env.dataset_name == 'clevrtex':
            print('using clevrtex variant suffix')
            return f"clevrtex_{self.dataset_variant.lower()}"
        elif env.dataset_name == 'clevr':
            return f"clevr_{self.dataset_variant.lower()}"
        else:
            print('dataset name error - dataset.py line 170')
            exit(0)

    def __init__(self,
                 path,
                 dataset_variant='full',
                 split='train',
                 crop=True,
                 resize=(128, 128),
                 return_metadata=True):
        self.return_metadata = return_metadata
        self.crop = crop
        self.resize = resize
        self.image_height = resize[0]
        self.image_width = resize[1]
        if dataset_variant not in self.variants:
            raise DatasetReadError(f"Unknown variant {dataset_variant}; [{', '.join(self.variants)}] available ")

        if split not in self.splits:
            raise DatasetReadError(f"Unknown split {split}; [{', '.join(self.splits)}] available ")
        if dataset_variant == 'outd':
            # No dataset splits in
            split = None

        self.dataset_variant = dataset_variant
        self.split = split
        print(f'path to dataset is {path}')
        self.basepath = Path(path)
        if not self.basepath.exists():
            raise DatasetReadError()
        sub_fold = self._variant_subfolder()
        if self.basepath.name != sub_fold:
            self.basepath = self.basepath / sub_fold
        #         try:
        #             with (self.basepath / 'manifest_ind.json').open('r') as inf:
        #                 self.index = json.load(inf)
        #         except (json.JSONDecodeError, IOError, FileNotFoundError):
        self.index, self.mask_index, self.metadata_index = self._reindex()

        print(f"Sourced {dataset_variant} ({split}) from {self.basepath}")

        bias, limit = self.splits.get(split, (0., 1.))
        if isinstance(bias, float):
            bias = int(bias * len(self.index))
        if isinstance(limit, float):
            limit = int(limit * len(self.index))
        self.limit = limit
        self.bias = bias

    def _format_metadata(self, meta):
        """
        Drop unimportanat, unsued or incorrect data from metadata.
        Data may become incorrect due to transformations,
        such as cropping and resizing would make pixel coordinates incorrect.
        Furthermore, only VBG dataset has color assigned to objects, we delete the value for others.
        """
        objs = []
        for obj in meta['objects']:
            o = {
                'material': obj['material'],
                'shape': obj['shape'],
                'size': obj['size'],
                'rotation': obj['rotation'],
            }
            if self.dataset_variant == 'vbg':
                o['color'] = obj['color']
            objs.append(o)
        return {
            'ground_material': meta['ground_material'],
            'objects': objs
        }

    def __len__(self):
        return self.limit - self.bias

    def __getitem__(self, ind):
        ind = self._index_with_bias_and_limit(ind)

        img = Image.open(self.index[ind])
        msk = Image.open(self.mask_index[ind])

        if self.crop:
            crop_size = int(0.8 * float(min(img.width, img.height)))
            img = img.crop(((img.width - crop_size) // 2,
                            (img.height - crop_size) // 2,
                            (img.width + crop_size) // 2,
                            (img.height + crop_size) // 2))
            msk = msk.crop(((msk.width - crop_size) // 2,
                            (msk.height - crop_size) // 2,
                            (msk.width + crop_size) // 2,
                            (msk.height + crop_size) // 2))
        if self.resize:
            img = img.resize(self.resize, resample=Image.BILINEAR)
            msk = msk.resize(self.resize, resample=Image.NEAREST)

        img = Ft.to_tensor(np.array(img)[..., :3])
        msk = torch.from_numpy(np.array(msk))[None]
        # added 255* because background model uses 0-255 inputs
        ret = (ind, 255.0*img, msk)

        if self.return_metadata:

            with self.metadata_index[ind].open('r') as inf:
                meta = json.load(inf)
            # added 255* because background model uses 0-255 inputs
            ret = (ind, 255.0*img, msk, self._format_metadata(meta))

        return ret


def collate_fn(batch):
    return (
      #  *torch.utils.data._utils.collate.default_collate([(b[0], b[1], b[2]) for b in batch]), [b[3] for b in batch])
      *torch.utils.data._utils.collate.default_collate([(b[0], b[1], b[2]) for b in batch]),)


def get_datasets(batch_size = env.batch_size):

    if env.dataset_name == "clevrtex" or env.dataset_name == 'clevr':

        train_dataset = CLEVRTEX(env.full_dataset_path, # Untar'ed
                dataset_variant='full', # 'full' for main CLEVRTEX, 'outd' for OOD, 'pbg','vbg','grassbg','camo' for variants.
                split='train',
                crop=True,
                resize=(env.image_height, env.image_width),
                return_metadata=False # Useful only for evaluation, wastes time on I/O otherwise
                )

        test_dataset = CLEVRTEX(env.full_dataset_path,  # Untar'ed
                                dataset_variant='full',
                                # 'full' for main CLEVRTEX, 'outd' for OOD, 'pbg','vbg','grassbg','camo' for variants.
                                # split='test',
                                split='test',
                                crop=True,
                                resize=(env.image_height, env.image_width),
                                return_metadata=False  # Useful only for evaluation, wastes time on I/O otherwise
                                )
    elif env.dataset_name == 'shapestacks':
        train_dataset = ShapeStacksDataset(mode='train',img_size=env.image_height)
        test_dataset = ShapeStacksDataset(mode='test', img_size=env.image_height)

    elif env.dataset_name == 'objects_room':
        train_dataset = Image_dataset(env.train_dataset_path,env.GT_train_dataset_path)
        test_dataset = Image_dataset(env.test_dataset_path,env.GT_test_dataset_path)

    else:
        assert env.train_dataset_path is not None , 'dataset path not defined'
        train_dataset = Image_dataset(env.train_dataset_path, env.GT_train_dataset_path)
        if env.test_dataset_path is None:
            print('warning : test dataset not defined, will use train dataset for tests')
            test_dataset = train_dataset
        else:
            test_dataset = Image_dataset(env.test_dataset_path, env.GT_test_dataset_path)

    if env.dataset_name in ["clevrtex"]:
        print('using collate_fn')
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=4,
                                                 drop_last=True, pin_memory=True,persistent_workers=True,
                                                 collate_fn=collate_fn)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                       shuffle=True, num_workers=4,
                                                       drop_last=True, pin_memory=True, persistent_workers=True,
                                                       collate_fn=collate_fn)

    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=env.batch_size,
                                                  num_workers=4,shuffle=True,
                                                  drop_last=True, pin_memory=True, persistent_workers=True,
                                                  )
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=env.batch_size,
                                                       num_workers=4, shuffle=True,
                                                       drop_last=True, pin_memory=True, persistent_workers=True,
                                                       )

    return train_dataset, train_dataloader, test_dataset, test_dataloader

