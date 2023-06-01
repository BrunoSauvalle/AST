from __future__ import print_function

import torch
import torch.optim as optim
import time
import os
import torch.utils.data
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

import utils
from dataset import  get_datasets
from generate_background_dataset import compute_background_and_mask_using_trained_model
from config import env

def background_loss(real_images,backgrounds_with_error_prediction):
    """ background loss used during training
        input images format : tensor shape N,3,H,W range 0-255"""

    bs, nc, h, w = real_images.size()

    backgrounds = backgrounds_with_error_prediction[:, 0:3, :, :]
    pixel_errors = torch.sum(torch.nn.functional.smooth_l1_loss(real_images, backgrounds,  reduction='none', beta=3.0), dim=1)*(1/255.0) # range 0-3

    error_prediction = backgrounds_with_error_prediction[:, 3, :, :]*(1/255) # range 0-1
    error_prediction_error =  torch.nn.functional.smooth_l1_loss(error_prediction, pixel_errors.detach()*(1/3),  reduction='none', beta=3.0/255)

    with torch.no_grad(): #  weights computation do not require gradient

        soft_masks = torch.tanh(pixel_errors*(1/env.tau_1)) # range 0-1
        weight_logit = -env.beta*torch.nn.functional.avg_pool2d(soft_masks, 2 * (w // env.r) + 1,
                                                         stride=1, padding= w // env.r, count_include_pad=False) # range 0-1 BSxHxW
        normalized_pixel_weights = torch.exp(weight_logit)*(1/(h*w*bs))

    loss = torch.sum((pixel_errors + error_prediction_error) * normalized_pixel_weights)

    return loss


def evaluate_background_complexity_using_trained_model(netBE, netBG):
    """evaluates whether the background changes are simple or complex using partially trained model"""

    batch_size = env.batch_size

    dataset, dataloader =  get_datasets()[:2]

    dataloader_iterator = iter(dataloader)
    if len(dataset) > batch_size * 15:
        number_of_batchs = 15
    else: # if number of frames <= 480, limit to one epoch
        number_of_batchs = len(dataset) // batch_size

    # placeholder for reconstructed backgrounds
    backgrounds_big_batch = torch.zeros(batch_size * number_of_batchs, 3, dataset.image_height, dataset.image_width)

    netBE.eval()
    netBG.eval()

    with torch.no_grad():
        for i in range(number_of_batchs):
            data = next(dataloader_iterator)[1].type(torch.cuda.FloatTensor)
            real_images = data
            backgrounds_with_error_predictions = netBG(netBE(real_images))
            backgrounds_big_batch[i * batch_size:(i + 1) * batch_size, :, :, :] = backgrounds_with_error_predictions[:, 0:3,
                                                                              :, :]

    median_background = torch.median(backgrounds_big_batch, dim=0, keepdim=True)[0].expand_as(backgrounds_big_batch)

    pixel_errors = (1 / 255) * torch.sum(
        torch.nn.functional.l1_loss(backgrounds_big_batch, median_background, reduction='none'), dim=1)

    soft_masks = torch.tanh(pixel_errors*(1/env.tau_1))
    average_mask_background_error = torch.mean(soft_masks)

    netBE.train()
    netBG.train()

    if average_mask_background_error > env.tau_0:
        complex_background = True
    else:
        complex_background = False

    return complex_background

def background_training_loop(outf,netBE, netBG,train_dataloader,optimizer,
                             number_of_steps, evaluation_step, background_complexity):


    writer = SummaryWriter()
    writetime = 0

    assert os.path.isdir(env.results_dir_path), f' directory {env.results_dir_path} does not exist'


    models_path = os.path.join(env.results_dir_path, "models")
    if not os.path.isdir(models_path):
        print(f'warning : models path {models_path} is not a directory, creating directory')
        os.mkdir(models_path)
    print(f'starting autoencoder training loop')

    netBE.train()
    netBG.train()

    saved_network = False
    save_network = False

    learning_rate_reduction_step = (4 * number_of_steps) // 5
    learning_rate_is_reduced = False

    def lmbda(epoch):
        return 0.1
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

    last_message_time = time.time()
    last_image_save_time = time.time()

    step = 0
    last_step = 0
    epoch = 0


    while True:

        for j, data in enumerate(train_dataloader, 0):

            indices, images, masks = data # indices : integer, images : NCHW 0-255, masks : NHW integer

            images = images.to(env.device)
            masks = masks.to(env.device)

            optimizer.zero_grad()

            encoder_latents = netBE(images)
            backgrounds_with_error_prediction = netBG(encoder_latents)  # range 0-255
            loss = background_loss(images, backgrounds_with_error_prediction)
            loss.backward()


            optimizer.step()

            writer.add_scalar('background_loss', loss, global_step=writetime)

            writetime = writetime + 1

            if step > number_of_steps and saved_network == False:
                save_network = True

            if saved_network == True:
                print(f'training finished ')
                return netBE, netBG, background_complexity

            if time.time() - last_message_time > 15 :
                with torch.no_grad():
                    background_masks = (masks == 0).to(int)
                    masked_differences = background_masks * (
                                images - backgrounds_with_error_prediction[:, 0:3, :, :]) / 255
                    L2_masked_error = torch.mean(torch.square(masked_differences))
                writer.add_scalar('L2_masked_error', L2_masked_error, global_step=writetime)

                its = (step - last_step) / (time.time()- last_message_time)
                last_message_time = time.time()
                last_step = step

                print('[dataset %s][epoch %d][step %d/%d it/s %.3f]  [loss: %.6f   L2 error on background segment %.6f]'
                      % (env.dataset_name, epoch, step, number_of_steps,its,
                         loss, L2_masked_error))
            if time.time() - last_image_save_time > 120 :
                print(f'saving training images in directory {outf}')
                last_image_save_time = time.time()

                with torch.no_grad():
                    output_images = compute_background_and_mask_using_trained_model( netBE, netBG, images)

                    predicted_foreground_masks = output_images['masks']/255 # numpy array
                    background_masks = (masks == 0).to(int)
                    concatenated_images = torch.cat([images[0:8]/255,backgrounds_with_error_prediction[0:8,0:3,:,:]/255,
                                                     1-background_masks[0:8].repeat(1,3,1,1),
                                                     torch.from_numpy(predicted_foreground_masks[0:8]).to(env.device).unsqueeze(1).repeat(1,3,1,1)])
                    vutils.save_image(concatenated_images,
                                      '%s/result_%d.png' % (outf, epoch), nrow=8)

                    mask_error = torch.mean(torch.square(background_masks.cpu()
                                                         - (1 - torch.from_numpy(predicted_foreground_masks).cpu()).unsqueeze(1)))
                    print(f'mask error : {mask_error}')
                    writer.add_scalar('mask_error', mask_error, global_step=writetime)
            step += 1

            if step == evaluation_step and env.unsupervised_mode:

                background_complexity = evaluate_background_complexity_using_trained_model(netBE, netBG)

                if background_complexity:
                    print('complex background detected, aborting current training and starting new training with updated model ')
                    return netBE, netBG, background_complexity
                else:
                    print('simple background, finishing training')

            if (step == 1000 or step % 20000 == 0) :
                model_path = os.path.join(models_path, f'trained_model_{step}.pth')
                utils.save_trained_model(netBE, netBG, optimizer, background_complexity, model_path)
                utils.save_trained_model(netBE, netBG, optimizer, background_complexity, '%s/model_%d.pth' % (outf, epoch % 2))

            if save_network == True:
                model_path = os.path.join(models_path, 'trained_model.pth')
                utils.save_trained_model(netBE, netBG, optimizer, background_complexity, model_path)
                utils.save_trained_model(netBE, netBG, optimizer, background_complexity, '%s/model_final.pth' % outf)
                print('final model saved as : ')
                print('%s/model_final.pth' % outf)
                saved_network = True

            if step >= learning_rate_reduction_step and learning_rate_is_reduced == False:
                scheduler.step()
                print(f'learning rate is now reduced (step {step})')
                learning_rate_is_reduced = True

        epoch = epoch + 1


def train_dynamic_background_model(outf):
    """ training function for dynamic background"""

    if env.unsupervised_mode:
        number_of_steps = env.n_simple
        evaluation_step = env.n_eval
        background_complexity = False
    else:
        number_of_steps = env.n_iterations
        background_complexity = env.background_complexity
        evaluation_step = 1e10 # no evaluation in supervised mode

    train_dataset, train_dataloader = get_datasets()[:2]

    if env.use_trained_model:

        print(f'loading saved models from {env.saved_model_path}')
        checkpoint = torch.load(env.saved_model_path)
        encoder_state_dict = checkpoint['encoder_state_dict']
        generator_state_dict = checkpoint['generator_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        background_complexity = checkpoint['complexity']

    netBE, netBG = utils.setup_background_models(env.image_height,env.image_width,background_complexity)


    optimizer = optim.AdamW([{'params': netBG.parameters()}, {'params': netBE.parameters()}],
                           lr=env.learning_rate, betas=(0.90, 0.999), weight_decay=env.weight_decay)

    if env.use_trained_model:
        netBE.load_state_dict(encoder_state_dict)
        netBG.load_state_dict(generator_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        print('model succesfully loaded')

    else:
        print('starting training from zero')


    netBE, netBG, complex_background = background_training_loop(outf,netBE, netBG,train_dataloader,optimizer,
                                                                                    number_of_steps, evaluation_step,background_complexity)

    if env.unsupervised_mode and complex_background :  # if the background is complex, start new training with more complex model

        number_of_steps = max(env.n_complex, (len(train_dataset) // env.batch_size) * env.e_complex)
        print(f'number of steps will be {number_of_steps}')
        evaluation_step = 1e10  # no evaluation

        netBE, netBG = utils.setup_background_models(env.image_height,env.image_width,complex_background,
                                                 )

        optimizer = optim.AdamW([{'params': netBG.parameters()}, {'params': netBE.parameters()}],
                                lr=env.learning_rate, betas=(0.90, 0.999), weight_decay=env.weight_decay)

        netBE, netBG, _ = background_training_loop(outf,netBE, netBG,train_dataloader,optimizer, number_of_steps, evaluation_step, background_complexity)

    return netBE, netBG

if __name__ == "__main__":

        outf, now = utils.setup_archive()
        if env.train_model:
            train_dynamic_background_model(outf)
        else:
            print('train_model is not authorized on this dataset')
