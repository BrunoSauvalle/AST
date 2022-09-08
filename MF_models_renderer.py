



import torch.nn.functional as F
import torch
import torch.nn as nn

from MF_config import args

def get_object_generator():
    if args.image_width <= 64 and args.image_height <= 64:
        print('object generator produces 32x32 images')
        return nn.Sequential(
            nn.ConvTranspose2d(args.z_what_dim, 64, 2, 1, 0, bias=False),
            # 2x2
            nn.GroupNorm(4, 64), nn.CELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            # 4x4
            nn.GroupNorm(2, 32), nn.CELU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            # 8x8
            nn.GroupNorm(1, 16), nn.CELU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1, bias=False),
            # 16x16
            nn.GroupNorm(1, 8), nn.CELU(),
            nn.ConvTranspose2d(8, 4, 4, 2, 1, bias=True),
            # 32x32
            nn.Sigmoid()
        )
    elif args.image_width <= 128 and args.image_height <= 128:

        print('object generator produces 64x64 images')

        return nn.Sequential(

            nn.ConvTranspose2d(args.z_what_dim, 128, 2, 1, 0, bias=False),
            # 2x2
            nn.GroupNorm(8, 128), nn.CELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            # 4x4
            nn.GroupNorm(4, 64), nn.CELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            # 8x8
            nn.GroupNorm(2, 32), nn.CELU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            # 16x16
            nn.GroupNorm(1, 16), nn.CELU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1, bias=False),
            # 32x32
            nn.GroupNorm(1, 8), nn.CELU(),
            nn.ConvTranspose2d(8, 4, 4, 2, 1, bias=True),
            # 64x64
            nn.Sigmoid()
        )
    else:
        print('object generator produces 128x128 images')

        return nn.Sequential(
            nn.ConvTranspose2d(args.z_what_dim, 256, 2, 1, 0, bias=False),
            # 2x2
            nn.GroupNorm(16, 256), nn.CELU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            # 4x4
            nn.GroupNorm(8, 128), nn.CELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            # 8x8
            nn.GroupNorm(4, 64), nn.CELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            # 16x16
            # state size. (ncg*2) x 4 x 4
            nn.GroupNorm(2, 32), nn.CELU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            # 32x32
            nn.GroupNorm(1, 16), nn.CELU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1, bias=False),
            # 64x64
            nn.GroupNorm(1, 8), nn.CELU(),
            nn.ConvTranspose2d(8, 4, 4, 2, 1, bias=True),
            # 128x128
            nn.Sigmoid()
        )


class Renderer(nn.Module):


    def __init__(self,args):
        super().__init__()

        self.args = args
        self.glimpse_generator = get_object_generator()
        self.scaling_dim = 1 if args.isotropic_scaling else 2
        self.background_activation_logit = nn.Parameter(args.initial_background_activation_logit*torch.ones(1,1), requires_grad = True)
        if args.variable_background_activation:
            self.local_background_activation_logit = nn.Parameter(torch.zeros(1, 1, 1, args.image_height, args.image_width), requires_grad=True)
        self.background_mask_logit = nn.Parameter(torch.zeros(1,1,1,1,1), requires_grad = False)
        self.background_mask = nn.Parameter(torch.ones(1, 1, 1, 1, 1), requires_grad=False)
        print('Generator created')

    def forward(self, latents, backgrounds):
        # computes scene using reconstructed background and object latents
        # latents shape should be batch_size x number of objects in batch x 2+(1 or 2)+1+z_what_dim
        # latents structure : 2d : position latents
        #                     1d or 2d : scaling factor logit
        #                     1d  : activation logit
        #                     z_what_dim d : z_what
        # backgrounds shape should be batch_size x 3 x h x w

        args = self.args

        h = args.image_height
        w = args.image_width
        z_what_dim = args.z_what_dim
        k = args.max_set_size
        n = latents.shape[1]

        query_activation_logit = latents[:, :, 4]

        n_objects = n * k # total number of objects in batch
        latents = latents.reshape(n_objects,z_what_dim+3+self.scaling_dim)
        position_latents = torch.clamp(latents[:, 0:2], min = -1, max = 1)
        scaling_factors = args.min_scaling_factor + (args.max_scaling_factor - args.min_scaling_factor) * torch.sigmoid( latents[ :, 2:2+self.scaling_dim])
        z_whats = latents[:, 3+self.scaling_dim:]

        # computation of object images  using the position latents
        object_images_with_masks = self.glimpse_generator(z_whats.reshape(n_objects,z_what_dim,1,1))

        # preparation of affine matrix theta for spatial transformer network
        theta = torch.zeros(n_objects, 2, 3, device = self.args.device)
        theta[:,0,0] = scaling_factors[:,0]
        theta[:,1,1] = scaling_factors[:,0] if args.isotropic_scaling else  scaling_factors[:,1]
        theta[:, :, 2] = -position_latents*scaling_factors

        #### computation of warped images using STN
        grid = torch.nn.functional.affine_grid(theta,torch.Size((n_objects, 4, h, w)))
        warped_rgba_images = torch.nn.functional.grid_sample(object_images_with_masks, grid).reshape(k,n,4,h,w)

        #### computation of complete images
        warped_rgb_images = warped_rgba_images[:,:,:3,:,:]
        warped_masks = warped_rgba_images[:,:,3,:,:].unsqueeze(2)

        query_activations = torch.exp(query_activation_logit).reshape(k,n,1,1,1).expand(k, n,
                                                                                           1, h, w)
        if args.variable_background_activation:
            background_activations = torch.exp(self.background_activation_logit.reshape(1,1,1,1,1).expand(1,n,1,h,w)+self.local_background_activation_logit.expand(1,n,1,h,w))
        else:
            background_activations = torch.exp(
                self.background_activation_logit.reshape(1, 1, 1, 1, 1).expand(1, n, 1, h, w))

        activations = torch.cat([background_activations,query_activations ], dim = 0)

        warped_masks_including_background = torch.cat( [self.background_mask.expand(1,n, 1, h, w), warped_masks], dim=0)  # K+1, BS, 1, h, w
        weights = activations * warped_masks_including_background # K+1, BS, 1, h, w
        inverse_normalization_factor = torch.reciprocal(torch.sum(weights, dim = 0, keepdim=True)).expand(k+1, -1, -1, -1, -1)
        normalized_weights = weights*inverse_normalization_factor
        foreground_masks = 1 - normalized_weights[0] # the complement of background mask is foreground mask
        image_layers_including_background = torch.cat([backgrounds.reshape(1,n,3,h,w),warped_rgb_images], dim=0) # K+1, BS,3,H,W
        weighted_image_layers = image_layers_including_background*normalized_weights.expand(-1,-1,3,-1,-1)
        reconstructed_images = torch.sum(weighted_image_layers, dim=0) # BS,3,H,W

        return reconstructed_images,foreground_masks, weighted_image_layers,normalized_weights
        #  batch_size,nc_output,h,w ;  max_set_size, batch_size,nc_output,h,w ;

