
from __future__ import print_function

import torch
from pytorch_model_summary import summary
import os
import datetime

import  models
from config import env


def setup_archive():
    #project_root = '/workspace/PycharmProjects/SCOD/MOS/MF/background'
    #root_arch = '/workspace/PycharmProjects/SCOD/MOS/MF/background/background_outputs'
    #root_arch = env.training_images_output_directory
    now = datetime.datetime.now()
    now = now.isoformat()
    outf = os.path.join(env.training_images_output_directory, now)

    try:
        os.makedirs(outf)
        print(f'training images are stored in directory {outf}')
    except OSError:
        print(f'warning : cannot create directory {outf}')
        pass

    # autosave
    #shutil.copyfile(os.path.join(project_root, 'dataset.py'), os.path.join(outf, 'dataset-arch.py'))
    #shutil.copyfile(os.path.join(project_root, 'generate_background_dataset.py'), os.path.join(outf, 'generate_images-arch.py'))
    #shutil.copyfile(os.path.join(project_root, 'train.py'), os.path.join(outf, 'train-arch.py'))
    #shutil.copyfile(os.path.join(project_root, 'utils.py'), os.path.join(outf, 'utils-arch.py'))
    #shutil.copyfile(os.path.join(project_root, 'models.py'), os.path.join(outf, 'models-arch.py'))
    #shutil.copyfile(os.path.join(project_root, 'config.py'), os.path.join(outf, 'config-arch.py'))

    return outf,now

def setup_background_models(image_height, image_width,complexity = False, test_model = True):
    """creates, tests and return background model before training  """


    torch.backends.cudnn.benchmark = True

    netBE = models.Background_Encoder(image_height, image_width,  complexity)
    netBG = models.Background_Generator(image_height, image_width, complexity)

    netBE.eval()
    netBG.eval()
    print(f'pushing models on device {env.device}')
    netBE.to(env.device)
    netBG.to(env.device)

    if  test_model :
        test_image = torch.zeros((1, 3, image_height, image_width)).to(env.device)
        background_latents = netBE(test_image)
        background_test = netBG(background_latents)
        print(f'description background encoder')
        print(summary(netBE, test_image.to(env.device), show_input=False))
        print(f'description background generator')
        background_latents = netBE(test_image.to(env.device))
        print(summary(netBG, background_latents, show_input=False))

    return netBE, netBG

def save_trained_model(netBE,netBG,optimizer,complexity,model_path):
    torch.save({'optimizer_state_dict': optimizer.state_dict(), 'encoder_state_dict': netBE.state_dict(),
                'generator_state_dict': netBG.state_dict(), 'complexity':complexity
                }, model_path)

def get_trained_model():

        model_path = env.saved_model_path
        checkpoint = torch.load(model_path)
        complexity = checkpoint['complexity']
        netBE, netBG = setup_background_models(env.image_height, env.image_width,
                                                     complexity)
        print(f'loading saved models from {model_path}')

        generator_state_dict = checkpoint['generator_state_dict']
        encoder_state_dict = checkpoint['encoder_state_dict']
        netBG.load_state_dict(generator_state_dict)
        netBE.load_state_dict(encoder_state_dict)
        print('models succesfully loaded')

        return netBE, netBG






