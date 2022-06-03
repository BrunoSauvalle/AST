from __future__ import print_function

import os
import torch
import torch.utils.data
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image


import utils
import dataset
from config import env


# recommended options for speed optimization
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.benchmark = True


def compute_background_and_mask_using_trained_model(netBE, netBG, data):
    """ compute  backgrounds and masks from a batch of images using trained model
    data should be a batch of sample range 0-255"""

    images = {}
    real_images = data.to(env.device).type(torch.cuda.FloatTensor)  # range 0-255 Nx3xHxW RGB
    backgrounds_with_error_predictions = netBG(netBE(real_images))  # range 0-255 shape Nx4xHxW RGB+error prédiction
    backgrounds = backgrounds_with_error_predictions[:, 0:3, :, :]
    error_predictions = backgrounds_with_error_predictions[:, 3, :, :]  # NHW  0-255 float

    batch_size = data.shape[0]

    diffs = (torch.abs((real_images - backgrounds))).permute(0, 2, 3, 1).cpu().detach().numpy()  # NHW3 RGB
    l1_errors = (0.333 * np.sum(diffs, axis=3)).astype('uint8')  # NHW range 0-255
    images['l_1'] = l1_errors

    error_predictions = error_predictions.cpu().detach().numpy().astype('uint8')  # NHW 0-255
    images['noise'] = error_predictions

    NWC_backgrounds_with_error_predictions = backgrounds_with_error_predictions.permute(0, 2, 3, 1).cpu().detach().numpy().astype('uint8')

    images['backgrounds_with_error_prediction'] = NWC_backgrounds_with_error_predictions

    NWC_backgrounds = backgrounds.permute(0, 2, 3, 1).cpu().detach().numpy().astype('uint8')  # NHW3 RGB
    backgrounds_opencv_format = NWC_backgrounds[:, :, :, ::-1]  # NHWC BGR
    images['backgrounds'] = backgrounds_opencv_format
    images['real'] = real_images.permute(0, 2, 3, 1).cpu().detach().numpy().astype('uint8')[:, :, :, ::-1]

    # placeholders for masks
    masks_before_post_processing = np.zeros((batch_size, env.image_height, env.image_width))
    masks = np.zeros((batch_size, env.image_height, env.image_width))

    for i in range(batch_size):

        corrected_dif = (np.maximum(0, l1_errors[i].astype('int16') - env.alpha_2 * error_predictions[i].astype('int16')))

        background_illumination = (torch.sum(backgrounds[i]) / (3 * env.image_height * env.image_width)).detach().cpu().numpy() # range 0-255

        mask_before_post_processing = 255 * np.greater(3*corrected_dif, env.alpha_1 * background_illumination).astype('uint8')

        masks_before_post_processing[i, :, :] = mask_before_post_processing

        if env.image_height < 400 and env.image_width < 400 : # no postprocessing on small images
            mask = mask_before_post_processing
        else:
            close_kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask_before_post_processing, cv2.MORPH_CLOSE, close_kernel)
            open_kernel = np.ones((7, 7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

        masks[i, :, :] = mask

    images['thresholded'] = masks_before_post_processing
    images['masks'] = masks
    return images


def generate_images(dataloader,video_paths, netBE,netBG):

                print(f"generating images...")
                CMAP = np.array([
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

                PALIMG = Image.new('P', (16, 16))
                PALIMG.putpalette(CMAP.flatten().tolist() * 4)

                for i, data in enumerate(tqdm(dataloader)):

                        indices, test_images, GT_masks = data
                        images = compute_background_and_mask_using_trained_model(netBE, netBG, test_images)

                        for j in range(test_images.shape[0]):
                            index = 1+i*env.batch_size+j

                            if GT_masks is not None:
                                GT_mask = GT_masks[j]

                            real_image = images['real'][j]
                            cv2.imwrite('%s/input_image_%06d.png' % (video_paths['input_images'], index),
                                        real_image)

                            background_with_error_prediction = images['backgrounds_with_error_prediction'][j]
                            background_with_error_prediction = cv2.cvtColor(
                                background_with_error_prediction, cv2.COLOR_RGBA2BGRA)
                            cv2.imwrite('%s/background_%06d.png' % (video_paths['backgrounds_with_error_prediction'], index),
                                background_with_error_prediction)

                            if GT_masks is not None:
                                GT_mask = GT_mask.reshape(env.image_height,env.image_width).detach().numpy().astype('uint8')
                                GT_mask = Image.fromarray(GT_mask, mode = 'P')
                                GT_mask.putpalette(CMAP.flatten().tolist() * 4)
                                GT_mask.save('%s/GT_mask_%06d.png' % (video_paths['GT_masks'], index))

                            cv2.imwrite('%s/background_%06d.png' % (video_paths['backgrounds'], index),
                                        images['backgrounds'][j])
                            cv2.imwrite('%s/bin%06d.png' % (video_paths['masks'], index), images['masks'][j])

def generate_images_using_trained_model():
    video_paths = {}
    results_path = env.results_dir_path
    train_dataset, train_dataloader, test_dataset, test_dataloader = dataset.get_datasets()
    model_path = env.saved_model_path
    print(f'loading saved models from {model_path}')
    checkpoint = torch.load(model_path)
    encoder_state_dict = checkpoint['encoder_state_dict']
    generator_state_dict = checkpoint['generator_state_dict']
    if "complexity " in checkpoint:
        complexity = checkpoint['complexity']
    else:
        complexity = True

    print(f'complexity = {complexity}')

    netBE, netBG = utils.setup_background_models(test_dataset.image_height, test_dataset.image_width,
                                                 complexity)
    netBE.load_state_dict(encoder_state_dict)
    netBG.load_state_dict(generator_state_dict)
    print('models succesfully loaded')

    video_paths['GT_masks'] = os.path.join(results_path, 'GT_masks_train')
    video_paths['masks'] = os.path.join(results_path, 'masks_train')
    video_paths['input_images'] = os.path.join(results_path, 'input_images_train')
    video_paths['backgrounds'] = os.path.join(results_path, 'background_images_train')
    video_paths['backgrounds_with_error_prediction'] = os.path.join(results_path, 'backgrounds_rgba_train')

    for k,dir_path in video_paths.items():
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
    print('starting background dataset generation on train dataset')
    generate_images(train_dataloader, video_paths, netBE, netBG)

    video_paths['GT_masks'] = os.path.join(results_path, 'GT_masks_test')
    video_paths['masks'] = os.path.join(results_path, 'masks_test')
    video_paths['input_images'] = os.path.join(results_path, 'input_images_test')
    video_paths['backgrounds'] = os.path.join(results_path, 'background_images_test')
    video_paths['backgrounds_with_error_prediction'] = os.path.join(results_path,
                                                                    'backgrounds_rgba_test')
    for k,dir_path in video_paths.items():
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
    print('starting background dataset generation on test dataset')
    generate_images(test_dataloader, video_paths, netBE, netBG)

if __name__ == "__main__":
        generate_images_using_trained_model()




