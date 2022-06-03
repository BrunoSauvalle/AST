
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import cv2
import torch.utils.data as data
import natsort
from PIL import Image

from MF_config import args

def get_loader(dataset, batch_size, num_workers=8, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )

"""
class Image_dataset(data.Dataset):

    def __init__(self, dataset_path):

        if os.path.exists(os.path.join(dataset_path, 'input')):
            self.dir = os.path.join(dataset_path, 'input')
        else:
            self.dir = dataset_path

        image_names = [item for item in os.listdir(self.dir) if os.path.isfile(os.path.join(self.dir, item))]
        image_names = natsort.natsorted(image_names)

        self.dataset_length = len(image_names)
        self.image_names = image_names

        first_image = self[0][0]
        nc_input, self.image_height, self.image_width = first_image.shape
        assert nc_input == 3 , f" input image with {nc_input} channels detected, input images should have 3 channels,"

        print(f'dataset initialized  w = {self.image_width},h = {self.image_height} number of frames {self.dataset_length}')

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
            image_path = os.path.join(self.dir, self.image_names[idx])
            opencv_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 0-255 HWC BGR
            rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)  # 0-255 HWC RGB
            np_image = np.asarray(rgb_image)
            np_image = np.transpose(np_image, (2, 0, 1))  # 0-255 CHW RGB
            tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor)/255
            return tensor_image, tensor_image"""

class Dual_Image_dataset(data.Dataset):
    # dataset providing input image, background image, and ground truth masks if available

    def __init__(self, input_dataset_path, background_dataset_path,GT_mask_dataset_path=None, input_nc=3,target_nc=3):

        if os.path.exists(os.path.join(input_dataset_path, 'input')):
            self.input_dir = os.path.join(input_dataset_path, 'input')
        else:
            self.input_dir = input_dataset_path

        if os.path.exists(os.path.join(background_dataset_path, 'input')):
            self.background_dir = os.path.join(background_dataset_path, 'input')
        else:
            self.background_dir = background_dataset_path

        if GT_mask_dataset_path is not None:
            if os.path.exists(os.path.join(GT_mask_dataset_path, 'input')):
                self.GT_mask_dir = os.path.join(GT_mask_dataset_path, 'input')
            else:
                self.GT_mask_dir = GT_mask_dataset_path

        input_image_names = [item for item in os.listdir(self.input_dir)
                             if os.path.isfile(os.path.join(self.input_dir, item))]
        input_image_names = natsort.natsorted(input_image_names)

        background_image_names = [item for item in os.listdir(self.background_dir) if
                             os.path.isfile(os.path.join(self.background_dir, item))]
        background_image_names = natsort.natsorted(background_image_names)
        self.background_image_names = background_image_names

        if GT_mask_dataset_path == None:
            self.GT_available = False
        else:
            self.GT_available = True

            GT_mask_image_names = [item for item in os.listdir(self.GT_mask_dir) if
                                  os.path.isfile(os.path.join(self.GT_mask_dir, item))]
            GT_mask_image_names = natsort.natsorted(GT_mask_image_names)
            self.GT_mask_image_names = GT_mask_image_names

        self.dataset_length = len(input_image_names)
        assert self.dataset_length > 1 , 'input dataset is empty'
        self.input_image_names = input_image_names

        assert len(background_image_names) == self.dataset_length, f'error input dataset has length {self.dataset_length} but background dataset lenght is {len(background_image_names)}'

        if self.GT_available:
            assert len(
                GT_mask_image_names) == self.dataset_length, f'error input dataset has length {self.dataset_length} but GT mask dataset lenght is {len(GT_mask_image_names)}'

        print(f'dataset lenght {len(background_image_names)}')
        self.input_nc = input_nc
        self.target_nc = target_nc

        first_input_image, first_target_image, first_GT_mask_image = self[0]

        input_nc, self.image_height, self.image_width = first_input_image.shape
        target_nc, target_image_height, target_image_width = first_target_image.shape

        assert target_image_height == self.image_height
        assert target_image_width == self.image_width
        print(f'finput dataset has {self.input_nc} channels, target dataset has {self.target_nc} channels')
        print(f'dataset initialized  w = {self.image_width},h = {self.image_height} number of frames {self.dataset_length}')

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
            input_image_path = os.path.join(self.input_dir, self.input_image_names[idx])
            input_opencv_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)  # 0-255 HWC BGR
            input_opencv_image = cv2.cvtColor(input_opencv_image, cv2.COLOR_BGR2RGB)
            input_np_image = np.asarray(input_opencv_image)
            input_np_image = np.transpose(input_np_image, (2, 0, 1))  # 0-255 CHW RGB
            input_tensor_image = torch.from_numpy(input_np_image).type(torch.FloatTensor)/255 # RGBA

            background_image_path = os.path.join(self.background_dir, self.background_image_names[idx])
            background_opencv_image = cv2.imread(background_image_path, cv2.IMREAD_UNCHANGED)  # 0-255 HWC BGR
            background_rgb_image = cv2.cvtColor(background_opencv_image, cv2.COLOR_BGRA2RGBA)
            background_np_image = np.asarray(background_rgb_image)
            background_np_image = np.transpose(background_np_image, (2, 0, 1))  # 0-255 CHW RGB or RGBA
            background_tensor_image = torch.from_numpy(background_np_image).type(torch.FloatTensor) / 255

            if self.GT_available:
                GT_mask_image_path = os.path.join(self.GT_mask_dir, self.GT_mask_image_names[idx])
                GT_mask_opencv_image = Image.open(GT_mask_image_path )
                GT_mask_tensor_image = torch.from_numpy(np.array(GT_mask_opencv_image))[None]
            else:
                GT_mask_tensor_image = 0
            return input_tensor_image, background_tensor_image, GT_mask_tensor_image # RGB, RGBA, 2D range 0-1


def get_train_dataset_and_dataloader():

        print(
        f'creating train dataset using directories {args.train_dataset_input_path},{args.train_dataset_background_path},{args.train_dataset_GT_mask_path}')

        train_dataset = Dual_Image_dataset(args.train_dataset_input_path, args.train_dataset_background_path,
                                               args.train_dataset_GT_mask_path)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                      shuffle=True, num_workers=args.workers,
                                                      drop_last=True, pin_memory=True,
                                                      persistent_workers=True)

        return train_dataset, train_dataloader


def get_test_dataset_and_dataloader(batch_size = args.batch_size):
    print(
        f'creating test dataset using directories {args.test_dataset_input_path},{args.test_dataset_background_path},{args.test_dataset_GT_mask_path}')

    test_dataset = Dual_Image_dataset(args.test_dataset_input_path, args.test_dataset_background_path,
                                       args.test_dataset_GT_mask_path)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                   shuffle=True, num_workers=args.workers,
                                                   drop_last=True, pin_memory=True,
                                                   persistent_workers=True)

    return test_dataset, test_dataloader






