

from __future__ import print_function


import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import time
import warnings
from torch.utils.tensorboard import SummaryWriter

from MF_config import args
import MF_utils
import MF_data
import MF_stats
import torch.utils.data
import MF_models_encoder
import MF_models_renderer

def losses(reconstructed_images,input_images, backgrounds_with_error_predictions,activation_layers):
    # input images should be torch tensor range 0-1 NCHW

    n, nc, h, w = backgrounds_with_error_predictions.shape
    activation_layers = activation_layers.reshape(1 + args.max_set_size,n,h,w)

    if args.real_world_video:
        threshold = args.threshold_for_real_world_videos

    else:
        if args.fixed_background:
            threshold = args.threshold_for_fixed_backgrounds
        else:
            threshold = args.threshold_for_dynamic_backgrounds

    pixel_reconstruction_errors = torch.sum(torch.abs(reconstructed_images - input_images), dim=1) # NHW range 0-3
    reconstruction_loss = torch.mean(torch.square(torch.relu(pixel_reconstruction_errors - threshold))) # scalar

    pixel_entropy_loss = torch.mean(torch.square(torch.sum(activation_layers*torch.log(activation_layers+1e-20), dim=0)))

    average_layer_activation_per_image = torch.mean(activation_layers, dim=(2,3))
    object_entropy_loss = torch.mean(torch.square(torch.sum(average_layer_activation_per_image*torch.log(average_layer_activation_per_image+1e-20),dim=0)))

    return reconstruction_loss,  pixel_entropy_loss, object_entropy_loss #mask_loss

class Training_state:
    """object containing the status of training """

    def __init__(self):
        self.step = 0
        self.last_step = 0
        self.rate = 0
        self.number_of_training_steps = args.number_of_training_steps
        self.epoch = 0

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def rate_update(self):
        step = self.step
        if step < args.warmup:
            return args.learning_rate*min(1,(step/args.warmup)**2)
        elif step < self.number_of_training_steps*(0.90):
            return args.learning_rate
        else:
            return args.learning_rate*0.1

    def update(self, optimizer):
        "Update parameters and rate"
        self.step += 1
        rate = self.rate_update()
        if rate != self.rate:
            for p in optimizer.param_groups:
                    p['lr'] = rate
        self.rate = rate

def object_train(archive_path):

    torch.cuda.empty_cache()
    train_dataset, train_dataloader = MF_data.get_train_dataset_and_dataloader()

    netE = MF_models_encoder.Encoder(args).to(args.device)
    netG = MF_models_renderer.Renderer(args).to(args.device)

    if args.use_trained_model == True:
        checkpoint = torch.load(args.object_model_checkpoint_path)
        netE.load_state_dict(checkpoint['encoder_state_dict'])
        netG.load_state_dict(checkpoint['generator_state_dict'])

    trainer = Training_state()

    optimizer = optim.Adam([{'params': netG.parameters()}, {'params': netE.parameters()}], lr=trainer.rate,
                                   betas=(0.90, 0.98), eps= 1e-9,weight_decay=args.object_detection_weight_decay)

    if args.use_trained_model == True:
        print('loading optimizer state')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading training state')
        trainer.load_state_dict(checkpoint['trainer_state_dict'])
        print(f'training state {trainer.state_dict()}')

    writer = SummaryWriter()
    last_showtime = 0
    last_savetime = 0
    last_message_time = time.time()

    netE.train()
    netG.train()

    print('starting object detection training loop')

    while True:

        torch.cuda.empty_cache()

        for j, data in enumerate(train_dataloader):

            input_images, background_images_with_error_prediction = data[:2]
            batch_size = input_images.shape[0]
            input_images = input_images.type(torch.cuda.FloatTensor).to(args.device)
            background_images_with_error_prediction = background_images_with_error_prediction.type(torch.cuda.FloatTensor).to(args.device)
            background_images = background_images_with_error_prediction[:, :3, :, :]

            netE.zero_grad()
            netG.zero_grad()

            latents, weights = netE(input_images)[:2]
            reconstructed_images, foreground_masks,warped_images,activation_layers = netG(latents, background_images)
            reconstruction_loss, pixel_entropy_loss,objects_entropy_loss = losses(reconstructed_images,input_images, background_images_with_error_prediction, activation_layers)

            pixel_entropy_loss_warmup_ratio = min(1,trainer.step / args.pixel_entropy_loss_full_activation_step)**2

            objects_entropy_loss_warmup_ratio = min(1,trainer.step / args.objects_entropy_loss_full_activation_step)**2

            loss = reconstruction_loss + pixel_entropy_loss * args.pixel_entropy_loss_weight* pixel_entropy_loss_warmup_ratio+ objects_entropy_loss*args.objects_entropy_loss_weight*objects_entropy_loss_warmup_ratio

            loss.backward()
            trainer.update(optimizer)
            optimizer.step()

            writer.add_scalar('loss', loss, global_step=trainer.step)

            if time.time() > last_message_time+args.message_time:

                its = (trainer.step-trainer.last_step)/(time.time()-last_message_time)
                lr = trainer.rate

                mse_loss, mIoU, msc, scaled_sc, msc_fg, scaled_sc_fg, ari, ari_fg, number_of_active_heads,average_number_of_activated_heads = MF_stats.evaluate(data, netE, netG)

                print(f'[dataset {args.dataset_name} ] [ archive path {archive_path}]')
                print('[ep %d][stp %d/%d its %.2f ] [lr %.2e ] [ loss: %.2e rec_l %.2e, pixel_l %.2e objects_l %.2e ] [active heads %.2f avg activated heads %.2f] '
                      % (trainer.epoch, trainer.step, args.number_of_training_steps, its,lr,
                         loss,reconstruction_loss, pixel_entropy_loss, objects_entropy_loss,  number_of_active_heads, average_number_of_activated_heads))
                print(f'[mIoU %.3f  msc_fg %.3f  ari_fg %.3f mse %2.f]' % (mIoU, msc_fg, ari_fg, mse_loss))
                writer.add_scalar('mse_loss', mse_loss, global_step=trainer.step)
                writer.add_scalar('mIoU', mIoU, global_step=trainer.step)
                writer.add_scalar('msc_fg', msc_fg, global_step=trainer.step)
                writer.add_scalar('ari_fg', ari_fg, global_step=trainer.step)
                writer.add_scalar('pixel entropy loss', pixel_entropy_loss, global_step=trainer.step)
                writer.add_scalar('objects entropy loss',objects_entropy_loss, global_step=trainer.step)
                writer.add_scalar('active heads', number_of_active_heads, global_step=trainer.step)
                writer.add_scalar('average_number_of_activated_heads', average_number_of_activated_heads, global_step=trainer.step)
                writer.add_scalar('reconstruction_loss', reconstruction_loss, global_step=trainer.step)

                trainer.last_step = trainer.step
                last_message_time = time.time()


            if time.time() > last_showtime + args.show_time:
                last_showtime = time.time()
                torch.cuda.empty_cache()
                print('saving images')
                train_images  = MF_utils.build_train_images(data, netE,netG)
                n_images = min(batch_size,args.n_images_to_show)
                vutils.save_image(train_images,
                                  '%s/comparison_train_set_%01d.png' % (archive_path, trainer.epoch), nrow=n_images,
                                  pad_value=1)

            if  time.time() > last_savetime +args.save_time :
                last_savetime = time.time()
                print('saving networks')
                torch.save({
                    'encoder_state_dict': netE.state_dict(),
                    'generator_state_dict' : netG.state_dict(),
                    'trainer_state_dict': trainer.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, '%s/checkpoint_%d.pth' % (archive_path, trainer.epoch % 2 ))

            if trainer.step == args.number_of_training_steps:
                print('end of training')
                torch.save({
                    'encoder_state_dict': netE.state_dict(),
                    'generator_state_dict': netG.state_dict(),
                    'trainer_state_dict': trainer.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict()
                }, '%s/checkpoint_final_%d_epochs.pth' % (archive_path, trainer.epoch ))
                print('model saved in %s/checkpoint_final_%d_epochs.pth' % (archive_path, trainer.epoch ))
                return True

            if trainer.step == args.evaluation_step and pixel_entropy_loss < args.detection_threshold:
                print('detection process initialization failed, new initialization of detection process....  ')
                return False

        trainer.epoch = trainer.epoch + 1


if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    cudnn.benchmark = True

    archive_path, now = MF_utils.setup_archive(args.train_dataset_input_path)
    manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    print('starting training for object detection')
    training_finished = False
    while not training_finished :
        training_finished = object_train(archive_path)
    print('end of training')


