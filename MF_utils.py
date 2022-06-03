
from __future__ import print_function

import os
import datetime
import torch
import torch.utils.data
from pytorch_model_summary import summary
import seaborn


from MF_config import args
import MF_models_encoder
import MF_models_renderer


def setup_object_models():

        netE = MF_models_encoder.Encoder(args).to(args.device)
        netG = MF_models_renderer.Renderer(args).to(args.device)
        test_image = 0.5*torch.ones((args.batch_size,3,args.image_height, args.image_width)).to(args.device)
        encoder_output = netE(test_image)
        print("netE test ok")
        print(summary(netE, test_image, show_input=False))
        generator_output = netG(encoder_output[0], test_image)
        print(summary(netG,encoder_output[0], test_image,show_input=False))
        print("netG test ok")
        return netE,netG

def load_final_checkpoint(netE, netG,object_model_checkpoint_path = args.object_model_checkpoint_path):
    print(f'loading objects checkpoint from checkpoint {object_model_checkpoint_path}')
    checkpoint = torch.load(object_model_checkpoint_path)
    netE.load_state_dict(checkpoint['encoder_state_dict'])
    netG.load_state_dict(checkpoint['generator_state_dict'])
    print('objects detection final checkpoint loaded')

def get_trained_model():
    netE, netG = setup_object_models()
    load_final_checkpoint(netE, netG)
    return netE, netG

def normalize(x):
    # rescale the heatmaps between 0 and 255
    y = x - x.min()
    z = y / y.max()
    return z*255

def normalize_1(x):
    # rescale the heatmaps between 0 and 1
    y = x - x.min()
    z = y / y.max()
    return z

@torch.no_grad()
def build_train_images(data,netE,netG):

    training_mode = netE.training
    assert training_mode == netG.training
    netE.eval()
    netG.eval()


    input_images, background_images_with_error_prediction, GT_masks = data
    batch_size = input_images.shape[0]
    n_images = min(batch_size, args.n_images_to_show)
    input_images = input_images.type(torch.cuda.FloatTensor).to(torch.device(0))[:n_images]
    background_images_with_error_prediction = background_images_with_error_prediction.type(torch.cuda.FloatTensor).to(args.device)

    h = args.image_height
    w = args.image_width
    nc = 3
    max_set_size = args.max_set_size


    background_images_with_error_prediction = background_images_with_error_prediction[:n_images]
    background_images = background_images_with_error_prediction[:,:nc,:,:]

    if torch.sum(GT_masks) == 0:
        GT_masks_available = False
    else:
        GT_masks_available = True
        GT_masks = GT_masks[:n_images]

    # model inference
    latents,  attentions_and_feature_maps = netE(input_images)[:2]
    rgb_images,foreground_masks, image_layers,activation_layers = netG(latents, background_images)

    input_images = input_images.cpu()
    foreground_masks = foreground_masks.cpu()
    rgb_images = rgb_images.cpu()
    activation_layers = activation_layers.cpu()

    if GT_masks is not None:
        GT_masks = GT_masks.cpu()

    batch_size = input_images.shape[0]
    assert batch_size == n_images

    foreground_masks = foreground_masks.repeat(1, nc, 1, 1)

    expanded_activation_layers = activation_layers.reshape(max_set_size+1, n_images, 1, h, w).expand(
        max_set_size+1, n_images, nc, h, w)
    if args.color_palette == 'default':
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
            [217, 217, 217],
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
            [217, 217, 217],
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
            [217, 217, 217]
        ])

        colors = CMAP.reshape(40, 3, 1, 1).expand(40, 3, h, w) / 255
        colors = colors[:max_set_size+1] # +1 is for background
    elif args.color_palette == 'seaborn':
        palette = seaborn.color_palette(palette = 'hls',n_colors=max_set_size)
        colors = torch.tensor(palette).reshape(max_set_size,3,1,1).expand(max_set_size,3,h,w)
        colors = torch.cat([torch.zeros(1,3,h,w),colors], dim = 0) # background should be black
    else:
        print(f'color palette {args.color_palette} not implemented')
        exit(0)

    color_index = torch.argmax(expanded_activation_layers, 0).expand(n_images, 3, h, w).clone().cpu()  # n,nc,H,W

    if GT_masks_available:
        GT_segmentation = torch.gather(colors,0,GT_masks.repeat(1,3,1,1).type(torch.int64))

    segmentation = torch.gather(colors, 0, color_index)  # BS, 3, h, w

    if  GT_masks_available:
        images = torch.cat([input_images,
                            GT_segmentation,
                            rgb_images,
                            segmentation
                            ])
    else:
        images = torch.cat([input_images,
                            rgb_images,
                            segmentation
                            ])

    if training_mode:
        images = torch.cat([images,foreground_masks])
        netE.train()
        netG.train()
    else:
        images = torch.cat([images, torch.ones(batch_size, 3, h, w)])

    return images

def setup_archive(dataset_path):

    #project_root = '/workspace/PycharmProjects/SCOD/MOS/MF'
    root_arch = args.training_images_output_directory
    now = datetime.datetime.now()
    now = now.isoformat()
    outf = os.path.join(root_arch, now + dataset_path[-20:].replace('/', ''))
    try:
        os.makedirs(outf)
    except OSError:
        print(f'warning : cannot create directory {outf}')
        pass
    # autosave
    #shutil.copyfile(os.path.join(project_root, 'MF_models_encoder.py'), os.path.join(outf, 'MF_models_encoder-arch.py'))
    #shutil.copyfile(os.path.join(project_root, 'MF_models_renderer.py'), os.path.join(outf, 'MF_models_renderer-arch.py'))
    #shutil.copyfile(os.path.join(project_root, 'MF_data.py'), os.path.join(outf, 'MF_data-arch.py'))
    #shutil.copyfile(os.path.join(project_root, 'MF_train.py'), os.path.join(outf, 'MF_train-arch.py'))
    #shutil.copyfile(os.path.join(project_root, 'MF_utils.py'), os.path.join(outf, 'MF_utils-arch.py'))
    #shutil.copyfile(os.path.join(project_root, 'MF_config.py'), os.path.join(outf, 'MF_config-arch.py'))

    return outf,now



