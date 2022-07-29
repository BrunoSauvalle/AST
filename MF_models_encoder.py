
import torch
import torch.nn as nn
from MF_config import args

from transformers import SegformerModel, SegformerConfig, SegformerForSemanticSegmentation


class Segformer_model(nn.Module):

    def __init__(self,out_channels):
        super().__init__()
        segformer_model = SegformerModel.from_pretrained(args.pretrained_backbone_weights_reference)
        configuration = segformer_model.config
        configuration.num_labels = out_channels
        self.complete_model = SegformerForSemanticSegmentation(configuration)
        self.complete_model.segformer = segformer_model

    def forward(self, x):
        outputs = self.complete_model(x)
        return outputs.logits

def compute_output_paddings(image_height,image_width,n_layers):

    output_paddings = []  # output paddings value for transpose convolutions

    current_w = image_width
    current_h = image_height

    for i in range(n_layers):

        pad_w = 1
        pad_h = 1
        str_w = 2
        str_h = 2
        ker_w = 4
        ker_h = 4

        new_w = 1 + (current_w + 2 * pad_w - ker_w) // str_w
        output_pad_w = (current_w + 2 * pad_w - ker_w) % str_w

        new_h = 1 + (current_h + 2 * pad_h - ker_h) // str_h
        output_pad_h = (current_h + 2 * pad_h - ker_h) % str_h

        output_paddings.append((output_pad_h, output_pad_w))

        current_h = new_h
        current_w = new_w


    return output_paddings



class Downsample_and_Conv (nn.Module):
    # a block composed of two convolution layers, the first one with stride 2, the second one with residual connection
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.bloc1 =  nn.Sequential(
            nn.Conv2d(in_channels,out_channels , 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.CELU())
        self.bloc2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3,1,1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.CELU())

    def forward(self, x):
        x = self.bloc1(x)
        x = self.bloc2(x)+x
        return x

class Conv_and_Upsample(nn.Module):
    # a block composed of two convolution layers, the last one transpose conv with stride 2, the first one with residual connection
    def __init__(self,in_channels,out_channels, output_padding):
        super().__init__()
        self.bloc1 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels , 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.CELU())
        self.bloc2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4,2,1,output_padding= output_padding, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.CELU())
    def forward(self, x):
        x = self.bloc1(x)+x
        x = self.bloc2(x)
        return x

class Unet(nn.Module):

    def __init__(self, image_height, image_width,nch_in,nch_out):
        super(Unet, self).__init__()

        self.nch_out = nch_out

        if image_height < 200 and image_width < 200:
            nlayers = 5
            nch = [80, 128, 192, 256, 256, 256]
        else:
            nlayers = 6
            nch = [48, 64, 96, 128, 128, 256, 256]
        print(f'using Unet with {nlayers} layers')

        self.nlayers = nlayers

        output_paddings = compute_output_paddings(image_height, image_width, nlayers)
        print(f'output_paddings = {output_paddings}')

        self.first_feature_map = nn.Sequential(nn.Conv2d(nch_in, nch[0], 3, 1, 1, bias=False),
                      nn.BatchNorm2d(nch[0]),
                      nn.CELU())

        self.downsample_blocks = nn.ModuleList([Downsample_and_Conv(nch[i],nch[i+1]) for i in range(nlayers)])

        self.center_block = nn.Sequential(
                          nn.Conv2d(nch[nlayers],nch[nlayers] , 3, 1, 1, bias=False),
                          nn.BatchNorm2d(nch[nlayers]),
                         nn.CELU())

        self.upsample_blocks = nn.ModuleList([Conv_and_Upsample(2*nch[i+1],nch[i], output_paddings[i])
                          for i in range(nlayers)])


        self.last_skip_layer = nn.Sequential(
                          nn.Conv2d(2*nch[0],nch_out , 3, 1, 1, bias=False),
                          nn.BatchNorm2d(nch_out),
                          nn.CELU())
        self.last_residual_layer = nn.Conv2d(nch_out,nch_out ,3, 1, 1, bias=False)
        self.conv1x1 = nn.Conv2d(nch_out, nch_out, 1, bias=False)

    def forward(self, x):

            feature_maps = []
            x = self.first_feature_map(x)
            feature_maps.append(x)

            for i in range(self.nlayers):
                x = self.downsample_blocks[i](x)
                feature_maps.append(x)
            x = self.center_block(x)

            for i in reversed(range(0, self.nlayers )):
                x = self.upsample_blocks[i](torch.cat([x, feature_maps[i + 1]], dim=1))

            x = self.last_skip_layer(torch.cat([x, feature_maps[0]], dim=1))
            x = self.last_residual_layer(x)+x
            x = self.conv1x1(x)
            return x

class Encoder(nn.Module):

    def __init__(self,args):
        super(Encoder,self).__init__()

        self.args = args
        h = args.image_height
        w = args.image_width

        self.scaling_dim = 1 if args.isotropic_scaling else 2

        feature_and_attention_map_dim = self.args.max_set_size+self.args.z_what_dim + 1 + self.scaling_dim # +1 is for activation

        if args.feature_map_generator_name == "Unet":
            self.feature_and_attention_map_generator = Unet(args.image_height, args.image_width,3, feature_and_attention_map_dim)
        elif args.feature_map_generator_name == "Segformer":
            self.feature_and_attention_map_generator = Segformer_model(feature_and_attention_map_dim)
        else:
            print(f'feature map generator name {args.feature_map_generator_name} not recognized ')
            exit(0)

        self.feature_and_attention_map_generator = self.feature_and_attention_map_generator.to(args.device)

        test_image = torch.rand(1,3,args.image_height,args.image_width).to(args.device)
        bs,n_channels,self.feature_map_height,self.feature_map_width = self.feature_and_attention_map_generator(test_image ).shape
        fw = self.feature_map_width
        fh = self.feature_map_height
        print(f'feature map shape is {n_channels,fh,fw}')
        assert n_channels ==  feature_and_attention_map_dim

        # position encoding for position latents, range normalized on [-1,1]
        position_map = torch.zeros((2, fh, fw))
        for x in range(fw):
            for y in range(fh):
                position_map[0, y, x] = 2*(x / (fw-1) - 0.5)
                position_map[1, y, x] = 2*(y / (fh-1) - 0.5)
        self.position_map = nn.Parameter(position_map.reshape( 1, 2, fh,fw), requires_grad=False)

        self.feature_vector_embedding = nn.Linear(self.args.z_what_dim + 3+self.scaling_dim, args.transformer_dim) # K,x,y,alpha, scale

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=args.transformer_dim, nhead=args.transformer_nhead,
                                                     dim_feedforward=args.transformer_dim_feedforward
                                                     )

        self.transformer = nn.TransformerEncoder(transformer_encoder_layer, num_layers=args.transformer_nlayers)

        self.latents_projection = nn.Linear(args.transformer_dim, self.args.z_what_dim + 3+self.scaling_dim) # 3 is for  x,y,  activation

        print('encoder network created')

    def forward(self, input_images ):

            # input shape  is N3HW
            n = input_images.shape[0]
            k = self.args.max_set_size
            fh = self.feature_map_height
            fw = self.feature_map_width


            feature_and_attention_map = self.feature_and_attention_map_generator(input_images)

            attention_logits = feature_and_attention_map[:, :k,:, :]

            attention_weights = torch.softmax(attention_logits.reshape(n,k,fh*fw), dim = 2).reshape(n,k,fh,fw)

            position_and_feature_maps = torch.cat([self.position_map.expand(n,-1,-1,-1), feature_and_attention_map[:,k:,:,:] ], dim = 1) # BS, 5+f, h, w

            expanded_position_and_feature_maps = position_and_feature_maps.unsqueeze(1).expand(-1,k,-1, -1, -1)

            expanded_attention_weights = attention_weights.unsqueeze(2).expand(-1,-1,  self.args.z_what_dim + 3+self.scaling_dim, -1,-1)

            position_and_feature_latents = torch.sum(expanded_position_and_feature_maps*expanded_attention_weights, dim = (3,4)) #  batch_size,max_set_size,feature_dim+4

            position_and_feature_latents = position_and_feature_latents.permute(1, 0, 2) # K, N, f+5

            # latents update with transformer encoder

            transformer_input = self.feature_vector_embedding(position_and_feature_latents)
            updated_position_and_feature_latents = self.latents_projection(self.transformer(transformer_input))

            feature_and_attention_maps = torch.cat([attention_weights,position_and_feature_maps],dim = 1 )

            return updated_position_and_feature_latents,  feature_and_attention_maps
            # shape of latents : max_set_size, batch_size, feature_dim+4/5 # 2 channels positions+ 1/2 channel scale + 1 activation + featuure dim channels
            # shape of weights :  batch_size,max set size, h, w
            # shape of weights :  batch_size,max set size, h, w


