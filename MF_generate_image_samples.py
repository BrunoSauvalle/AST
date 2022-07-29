
from __future__ import print_function
import os
import torchvision.utils as vutils
import warnings

from MF_config import args
import MF_utils
import MF_data


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print('creating model..')
    netE, netG = MF_utils.setup_object_models()
    print('loading checkpoint..')
    MF_utils.load_final_checkpoint(netE, netG, object_model_checkpoint_path = args.object_model_checkpoint_path)
    print('creating  dataloader..')
    netE.eval()
    netG.eval()

    n_images = 20
    number_of_samples = 1
    div = 4


    test_dataset, test_dataloader = MF_data.get_test_dataset_and_dataloader(batch_size = n_images)

    for idx in range(number_of_samples):
        data = next(iter(test_dataloader))

        images = MF_utils.build_train_images(data, netE, netG, n_images)

        N,C,H,W = images.shape
        print(f'images shape is {images.shape}')
        images = images.reshape(N//div,div,C,H,W).permute(1,0,2,3,4).reshape(N,C,H,W)

        vutils.save_image(images,
                          os.path.join(args.results_dir_path, f"sample_{idx}.png"), nrow=n_images//div,
                          pad_value=1)
    print(f'image samples saved in directory {args.results_dir_path}')



