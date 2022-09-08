
import torch
import torch.utils.data


class Foreground_training_configuration_data:
        def __init__(self,dataset_name,kwargs):

            self.dataset_name = dataset_name # dataset name

            ## dataset and checkpoint Paths
            self.test_dataset_input_path = None # Path to the frames sequence directory used for testing
            self.test_dataset_background_path = None
            self.test_dataset_target_path = None # Path to the frames sequence directory used for testing
            self.test_dataset_GT_mask_path = None # Path to the ground truth masks directory used for evaluation during testing
            self.train_dataset_input_path = None  # Path to the frames sequence directory used for testing
            self.train_dataset_background_path = None
            self.train_dataset_target_path = None  # Path to the frames sequence directory used for testing
            self.train_dataset_GT_mask_path = None  # Path to the ground truth masks directory used for evaluation during testing
            self.object_model_checkpoint_path = None
            self.results_dir_path = None

            ## videos descriptors
            self.max_set_size = None
            self.image_height = None
            self.image_width = None


            # model training scenario
            self.background_model_training_scenario = "curriculum_training" # possible scenarii : "random_initialization", "frozen_weights", "curriculum_training"


            # model parameters
            self.feature_map_generator_name = "Segformer" # "Unet" or "Segformer"
            self.pretrained_backbone_weights_reference = "nvidia/mit-b3" # for Segformer cf Hugging Face website for available weights
            self.isotropic_scaling = None
            self.transformer_dim = 256 # dimension of inputs and outputs of the transformer encoder
            self.transformer_nhead = 8 # number of heads of  transformer encoder
            self.transformer_nlayers = 6 # number of layers of the transformer encoder
            self.transformer_dim_feedforward = 512 # number of layers of  transformer
            self.initial_background_activation_logit = 11
            self.variable_background_activation = False
            self.z_what_dim = None
            self.max_scaling_factor = None
            self.min_scaling_factor = 1.3

            # loss function parameters
            self.pixel_entropy_loss_weight =  1e-2

            # training parameters
            self.batch_size = None
            self.learning_rate = None
            self.number_of_training_steps = None
            self.warmup = None
            self.background_weights_freeze_duration = None
            self.pixel_entropy_loss_full_activation_step = None
            self.workers = 4 #number of workers for data loading per process
            self.object_detection_weight_decay = 0
            self.use_trained_model = False # if set to true, training will start from the saved checkpoint
            self.device = torch.device(0)

            # user interface parameters
            self.training_images_output_directory = '/workspace/PycharmProjects/SCOD/MOS/MF/MF_outputs' # to be updated
            self.n_images_to_show = 6 # number of images to show during training
            self.message_time = 15 # time between two training status message
            self.show_time = 60 # time between images saving (sec)
            self.save_time = 300 # time between model saving
            self.color_palette = None # default or seaborn

            # override default values with provided arguments:
            for attr, value in kwargs.items():
                assert hasattr(self,attr),f'wrong attribute name {attr}'
                setattr(self, attr, value)


synthetic_images_config = {
                        'z_what_dim': 32,
        'batch_size': 64, "learning_rate": 4e-5,
        'number_of_training_steps': 125000,
        "warmup" : 5000,
        "pixel_entropy_loss_full_activation_step": 10000,
        "background_weights_freeze_duration": 30000,
        "max_scaling_factor": 24,
        "variable_background_activation" : False,
        "color_palette" : "default"}

natural_images_config = {'image_height': 200, 'image_width': 320,

        'batch_size': 24, "learning_rate": 1e-5,
        'number_of_training_steps': 500000,
         "warmup" : 20000,
        "pixel_entropy_loss_full_activation_step" : 20000,
            "background_weights_freeze_duration" : 60000,
        "max_scaling_factor": 40,
        "isotropic_scaling" : False,
        "variable_background_activation" : True,
        "color_palette" : "seaborn"}

dataset_configs = {'clevrtex':
             {**synthetic_images_config,
                    'image_height': 128, 'image_width': 128, 'max_set_size': 10,"isotropic_scaling" : False,
                 "train_dataset_input_path": "/workspace/nvme0n1p1/Datasets/clevrtex/background_output/input_images_train",
                   "train_dataset_background_path": "/workspace/nvme0n1p1/Datasets/clevrtex/background_output/backgrounds_rgba_train",
                   "train_dataset_GT_mask_path": "/workspace/nvme0n1p1/Datasets/clevrtex/background_output/GT_masks_train",
                    "test_dataset_input_path": "/workspace/nvme0n1p1/Datasets/clevrtex/background_output/input_images_test",
                   "test_dataset_background_path": "/workspace/nvme0n1p1/Datasets/clevrtex/background_output/backgrounds_rgba_test",
                   "test_dataset_GT_mask_path": "/workspace/nvme0n1p1/Datasets/clevrtex/background_output/GT_masks_test",
                   #"object_model_checkpoint_path" :"/workspace/PycharmProjects/SCOD/MOS/MF/MF_outputs/2022-05-13T17:48:29.414474tinput_images_train/checkpoint_final_199_epochs.pth",
                    #"object_model_checkpoint_path":"/workspace/PycharmProjects/SCOD/MOS/MF/MF_outputs/2022-08-21T19:25:17.830730tinput_images_train/checkpoint_final_199_epochs.pth",
                   "object_model_checkpoint_path":"/workspace/PycharmProjects/SCOD/MOS/MF/MF_outputs/2022-09-08T16:30:49.586062tinput_images_train/checkpoint_0.pth",
                   "results_dir_path": "/workspace/nvme0n1p1/Datasets/clevrtex/results",
                    "use_trained_model": False,

                    },
            'ood':
             {**synthetic_images_config,
                    'image_height': 128, 'image_width': 128,
                    'max_set_size': 10,"isotropic_scaling" : False,
                    "test_dataset_input_path": "/workspace/nvme0n1p1/Datasets/clevrtex/outd/input_images_test",
                   "test_dataset_background_path": "/workspace/nvme0n1p1/Datasets/clevrtex/outd/backgrounds_rgba_test",
                   "test_dataset_GT_mask_path": "/workspace/nvme0n1p1/Datasets/clevrtex/outd/GT_masks_test",
              "object_model_checkpoint_path": "/workspace/PycharmProjects/SCOD/MOS/MF/MF_outputs/2022-05-13T17:48:29.414474tinput_images_train/checkpoint_final_199_epochs.pth",
                 "results_dir_path": "/workspace/nvme0n1p1/Datasets/clevrtex/results",
                    },
            'camo':
             {**synthetic_images_config,
                    'image_height': 128, 'image_width': 128,
                    'max_set_size': 10, "isotropic_scaling" : False,
                    "test_dataset_input_path": "/workspace/nvme0n1p1/Datasets/clevrtex/camo/input_images_test",
                   "test_dataset_background_path": "/workspace/nvme0n1p1/Datasets/clevrtex/camo/backgrounds_rgba_test",
                   "test_dataset_GT_mask_path": "/workspace/nvme0n1p1/Datasets/clevrtex/camo/GT_masks_test",
              "object_model_checkpoint_path": "/workspace/PycharmProjects/SCOD/MOS/MF/MF_outputs/2022-05-13T17:48:29.414474tinput_images_train/checkpoint_final_199_epochs.pth",
                 "results_dir_path": "/workspace/nvme0n1p1/Datasets/clevrtex/results",
              },


        'clevr':
            {**synthetic_images_config,
                    'image_height': 128, 'image_width': 128,
                    'max_set_size': 10,"isotropic_scaling" : True,
             "train_dataset_input_path": "/workspace/nvme0n1p1/Datasets/clevr/background_output/input_images_train",
             "train_dataset_background_path": "/workspace/nvme0n1p1/Datasets/clevr/background_output/backgrounds_rgba_train",
            "train_dataset_GT_mask_path": "/workspace/nvme0n1p1/Datasets/clevr/background_output/GT_masks_train",
             "test_dataset_input_path": "/workspace/nvme0n1p1/Datasets/clevr/background_output/input_images_test",
             "test_dataset_background_path": "/workspace/nvme0n1p1/Datasets/clevr/background_output/backgrounds_rgba_test",
             "test_dataset_GT_mask_path": "/workspace/nvme0n1p1/Datasets/clevr/background_output/GT_masks_test",


             "object_model_checkpoint_path":"/workspace/PycharmProjects/SCOD/MOS/MF/MF_outputs/2022-05-14T21:34:34.339958tinput_images_train/checkpoint_final_99_epochs.pth",
            "object_model_checkpoint_path_alt1":"/workspace/PycharmProjects/SCOD/MOS/MF/MF_outputs/2022-05-18T15:15:28.076594tinput_images_train/checkpoint_final_99_epochs.pth",
                   "results_dir_path": "/workspace/nvme0n1p1/Datasets/clevr/results",
                    "use_trained_model": False},
         'shapestacks':
             {**synthetic_images_config,
                'image_height': 64,'image_width':64,'max_set_size': 6,"isotropic_scaling" : True,
                 "train_dataset_input_path": "/workspace/nvme0n1p1/Datasets/shapestacks/background_output/input_images_train",
                   "train_dataset_background_path": "/workspace/nvme0n1p1/Datasets/shapestacks/background_output/backgrounds_rgba_train",
                   "train_dataset_GT_mask_path": "/workspace/nvme0n1p1/Datasets/shapestacks/background_output/GT_masks_train",
                    "test_dataset_input_path": "/workspace/nvme0n1p1/Datasets/shapestacks/background_output/input_images_test",
                   "test_dataset_background_path": "/workspace/nvme0n1p1/Datasets/shapestacks/background_output/backgrounds_rgba_test",
                   "test_dataset_GT_mask_path": "/workspace/nvme0n1p1/Datasets/shapestacks/background_output/GT_masks_test",
                "object_model_checkpoint_path_alt2":"",
                "object_model_checkpoint_path_alt1":"/workspace/PycharmProjects/SCOD/MOS/MF/MF_outputs/2022-05-18T11:56:10.372418tinput_images_train/checkpoint_final_36_epochs.pth",
                  "object_model_checkpoint_path": "/workspace/PycharmProjects/SCOD/MOS/MF/MF_outputs/2022-05-13T17:48:59.826827tinput_images_train/checkpoint_final_36_epochs.pth",
              "results_dir_path" :  "/workspace/nvme0n1p1/Datasets/shapestacks/results",
                "use_trained_model": False

                    },


         'objects_room':
             {**synthetic_images_config,
            'image_height': 64,'image_width':64,'max_set_size': 3,"isotropic_scaling" : False,
               "train_dataset_input_path": "/workspace/nvme0n1p1/Datasets/objects_room/background_output/input_images_train",
                   "train_dataset_background_path": "/workspace/nvme0n1p1/Datasets/objects_room/background_output/backgrounds_rgba_train",
                   "train_dataset_GT_mask_path": "/workspace/nvme0n1p1/Datasets/objects_room/background_output/GT_masks_train",
                    "test_dataset_input_path": "/workspace/nvme0n1p1/Datasets/objects_room/background_output/input_images_test",
                   "test_dataset_background_path": "/workspace/nvme0n1p1/Datasets/objects_room/background_output/backgrounds_rgba_test",
                   "test_dataset_GT_mask_path": "/workspace/nvme0n1p1/Datasets/objects_room/background_output/GT_masks_test",
                "object_model_checkpoint_path":"/workspace/PycharmProjects/SCOD/MOS/MF/MF_outputs/2022-07-29T10:15:42.491299tinput_images_train/checkpoint_0.pth",
                    "results_dir_path" :  "/workspace/nvme0n1p1/Datasets/objects_room/results",
                     },

    "southampton2":
       {**natural_images_config,
        'z_what_dim': 16, 'max_set_size': 24,
        "train_dataset_input_path": "/workspace/nvme0n1p1/Datasets/southampton2/background_output/input_images_train",
        "train_dataset_background_path": "/workspace/nvme0n1p1/Datasets/southampton2/background_output/backgrounds_rgba_train",
        "test_dataset_input_path": "/workspace/nvme0n1p1/Datasets/southampton2/background_output/input_images_train",
        "test_dataset_background_path": "/workspace/nvme0n1p1/Datasets/southampton2/background_output/backgrounds_rgba_train",
        "object_model_checkpoint_path": "/workspace/PycharmProjects/SCOD/MOS/MF/MF_outputs/2022-07-21T14:03:55.634360tinput_images_train/checkpoint_final_132_epochs.pth",
        "results_dir_path": "/workspace/nvme0n1p1/Datasets/southampton2/foreground_output/",
        "use_trained_model" : False
        },

    "trekell":
     {**natural_images_config,
        'z_what_dim': 24, 'max_set_size': 16,
        "train_dataset_input_path": "/workspace/nvme0n1p1/Datasets/trekell/background_output/input_images_train",
        "train_dataset_background_path": "/workspace/nvme0n1p1/Datasets/trekell/background_output/backgrounds_rgba_train",
        "test_dataset_input_path": "/workspace/nvme0n1p1/Datasets/trekell/background_output/input_images_train",
        "test_dataset_background_path": "/workspace/nvme0n1p1/Datasets/trekell/background_output/backgrounds_rgba_train",
        "object_model_checkpoint_path": "/workspace/PycharmProjects/SCOD/MOS/MF/MF_outputs/2022-05-21T15:26:01.413344tinput_images_train/checkpoint_final_133_epochs.pth",
        "results_dir_path": "/workspace/nvme0n1p1/Datasets/trekell/foreground_output/",

    },
    "varna3":
              {**natural_images_config,
                'z_what_dim': 24, 'max_set_size': 16,
                 "train_dataset_input_path": "/workspace/nvme0n1p1/Datasets/varna3/background_output/input_images_train",
                 "train_dataset_background_path": "/workspace/nvme0n1p1/Datasets/varna3/background_output/backgrounds_rgba_train",
               "test_dataset_input_path": "/workspace/nvme0n1p1/Datasets/varna3/background_output/input_images_train",
               "test_dataset_background_path": "/workspace/nvme0n1p1/Datasets/varna3/background_output/backgrounds_rgba_train",
               #  "object_model_checkpoint_path": "/workspace/PycharmProjects/SCOD/MOS/MF/MF_outputs/2022-05-03T10:59:31.094292tinput_images_train/checkpoint_final_114_epochs.pth",
               "object_model_checkpoint_path":     "/workspace/PycharmProjects/SCOD/MOS/MF/MF_outputs/2022-05-21T21:57:37.845459tinput_images_train/checkpoint_final_142_epochs.pth",
                 "results_dir_path": "/workspace/nvme0n1p1/Datasets/varna3/foreground_output/",
                "use_trained_model": False,
             },
    }

# synthetic datasets


#dataset_name = 'shapestacks'
dataset_name = 'objects_room'
#dataset_name = 'clevrtex'
#dataset_name = 'clevr'
#dataset_name = 'camo'
#dataset_name = 'ood'

# real-world datasets

#dataset_name = 'southampton2'
#dataset_name = 'trekell'
#dataset_name = 'varna3'


assert dataset_name in ['trekell','varna3','southampton2','clevrtex','camo','ood','clevr','shapestacks','objects_room']

args = Foreground_training_configuration_data(dataset_name,dataset_configs[dataset_name])

