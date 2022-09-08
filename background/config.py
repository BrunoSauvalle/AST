# configuration file for background model
# All path data should be updated according to the user's filesystem
# dataset_name to be updated with the name of the active dataset

import torch

class Background_training_configuration_data:

        def __init__(self, dataset_name, kwargs):

            self.dataset_name = dataset_name
            self.image_height = None
            self.image_width = None
            self.full_dataset_path = None
            self.train_dataset_path = None
            self.test_dataset_path = None
            self.GT_train_dataset_path = None # ground truth masks path
            self.GT_test_dataset_path = None
            self.results_dir_path = None
            self.use_trained_model = False
            self.saved_model_path = None
            self.unsupervised_mode = False # the number of iterations and background complexity have to be set manually
            self.n_iterations = None
            self.background_complexity = True
            self.train_model = True
            self.learning_rate = 2e-3
            self.weight_decay = 0
            self.batch_size = 128
            self.device = torch.device(0)

            # path where temporary training samples are stored during training
            self.training_images_output_directory = '/workspace/PycharmProjects/SCOD/MOS/MF/background/background_outputs'

            # standard AE-NE hyperparameters
            self.beta = 6
            self.r = 75
            self.tau_0 = 0.24
            self.tau_1 = 0.25
            self.alpha_1 = 96/255
            self.alpha_2 = 7.0

            # override default values with arguments provided at init:
            for attr, value in kwargs.items():
                assert hasattr(self, attr), f'wrong attribute name {attr}'
                setattr(self, attr, value)

background_config = { 'clevrtex':
        {"image_height": 128, "image_width": 128,"n_iterations": 500000,
           "full_dataset_path": '/workspace/nvme0n1p1/Datasets/clevrtex/clevrtex_full',
           "results_dir_path": "/workspace/nvme0n1p1/Datasets/clevrtex/background_output_new",
           "saved_model_path": "/workspace/PycharmProjects/SCOD/MOS/MF/background/background_outputs/2022-03-19T00:17:40.933167/model_final.pth",
           },
             'clevr':
        {"image_height": 128, "image_width": 128,"n_iterations": 2500,"background_complexity" : False,
           "full_dataset_path": '/workspace/nvme0n1p1/Datasets/clevr/clevr_full',
           "results_dir_path": "/workspace/nvme0n1p1/Datasets/clevr/background_output",
           "saved_model_path": "/workspace/PycharmProjects/SCOD/MOS/MF/background/background_outputs/2022-03-30T12:03:44.180265/model_final.pth",
           },
         'shapestacks':
     {"image_height": 64, "image_width": 64,  "n_iterations": 500000,
            "full_dataset_path": "/workspace/nvme0n1p1/Datasets/shapestacks",
           "results_dir_path": "/workspace/nvme0n1p1/Datasets/shapestacks/background_output",
           "saved_model_path": "/workspace/PycharmProjects/SCOD/AENE_outputs/2022-03-15T23:04:37.103648/model_final.pth"},

        'objects_room':
     {"image_height": 64, "image_width": 64,"n_iterations": 500000,
           "train_dataset_path": '/workspace/nvme0n1p1/Datasets/objects_room/input_train',
            "test_dataset_path": '/workspace/nvme0n1p1/Datasets/objects_room/input_test',
            "GT_train_dataset_path" : '/workspace/nvme0n1p1/Datasets/objects_room/GT_masks_train',
            "GT_test_dataset_path" : '/workspace/nvme0n1p1/Datasets/objects_room/GT_masks_test',
           "results_dir_path": "/workspace/nvme0n1p1/Datasets/objects_room/background_output",
           "saved_model_path": "/workspace/PycharmProjects/SCOD/MOS/MF/background/background_outputs/2022-03-25T15:03:00.262679/model_final.pth"},

        "southampton2":
            {"image_height": 200, "image_width": 320,"batch_size": 64, "n_iterations":10000,
           "train_dataset_path": "/workspace/nvme0n1p1/Datasets/southampton2/input_frames",
           "results_dir_path": "/workspace/nvme0n1p1/Datasets/southampton2/background_output",
           "saved_model_path": "/workspace/PycharmProjects/SCOD/MOS/MF/background/background_outputs/2022-04-25T19:02:31.612350/model_final.pth"},

        "trekell":
            {"image_height": 200, "image_width": 320,"batch_size": 64, "n_iterations": 10000,
           "train_dataset_path": "/workspace/nvme0n1p1/Datasets/trekell/input_frames",
           "results_dir_path": "/workspace/nvme0n1p1/Datasets/trekell/background_output",
           "saved_model_path": "/workspace/PycharmProjects/SCOD/MOS/MF/background/background_outputs/2022-04-25T19:05:14.744867/model_final.pth"},

        "varna3":
            {"image_height": 200, "image_width": 320, "batch_size": 64, "n_iterations": 10000,
           "train_dataset_path": "/workspace/nvme0n1p1/Datasets/varna3/input_frames",
           "results_dir_path": "/workspace/nvme0n1p1/Datasets/varna3/background_output",
           "saved_model_path": "/workspace/PycharmProjects/SCOD/MOS/MF/background/background_outputs/2022-04-27T21:43:47.743933/model_final.pth"}
                      }

dataset_name = 'objects_room'

env = Background_training_configuration_data(dataset_name,background_config[dataset_name])


