
from __future__ import print_function

import torch
import warnings
from tqdm import tqdm
from collections import defaultdict
import torch.utils.data

from MF_config import args
import MF_utils
import MF_data
import MF_stats

if args.background_model_training_scenario != "frozen_weights":
    import sys
    sys.path.insert(0, './background')
    from background.utils import get_trained_model
    BE,BG = get_trained_model(args.background_model_training_scenario)
    BE.eval()
    BG.eval()
else:
    BE = None
    BG = None

class RunningMean:
    def __init__(self):
        self.v = 0.
        self.n = 0

    def update(self, v, n=1):
        self.v += v * n
        self.n += n

    def value(self):
        if self.n:
            return self.v / (self.n)
        else:
            return float('nan')

    def __str__(self):
        return str(self.value())

stats = defaultdict(RunningMean)
tags = defaultdict(lambda: defaultdict(lambda: defaultdict(RunningMean)))

def add_statistic(name, value, **tags):
    n = 1
    if isinstance(value, torch.Tensor):
        value = value.cpu().detach()
        if len(value.shape):
            n = value.shape[0]
            value = torch.mean(value)
        value = value.item()
    stats[name].update(value, n)
    for k, v in tags.items():
        tags[name][k][v].update(value, n)


def statistic(name, tag=None):
    if tag is None:
        return stats[name].value()
    r = [(k, rm.value()) for k, rm in tags[name][tag].items()]
    r = sorted(r, key=lambda x: x[1])
    return r

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print('creating model..')
    netE, netG = MF_utils.setup_object_models()
    print('loading checkpoint..')
    MF_utils.load_final_checkpoint(netE, netG, object_model_checkpoint_path = args.object_model_checkpoint_path,background_encoder= BE, background_generator= BG)
    print('creating  dataloader..')
    netE.eval()
    netG.eval()
    test_dataset, test_dataloader = MF_data.get_test_dataset_and_dataloader()
    print('starting evaluation..')
    with torch.no_grad():
        for j, data in enumerate(tqdm(test_dataloader)):

            mse_loss, mIoU, msc, scaled_sc, msc_fg, scaled_sc_fg,  ari, ari_fg, number_of_active_heads, average_number_of_activated_heads = MF_stats.evaluate(data, netE,
                                                                                                                                                                           netG, reduction=False, background_encoder=BE, background_generator=BG)
            add_statistic('mse_loss',mse_loss)
            add_statistic('mIoU', mIoU)
            add_statistic('msc',msc )
            add_statistic('msc_fg',msc_fg )
            add_statistic('scaled_sc',scaled_sc )
            add_statistic('scaled_sc_fg',scaled_sc_fg )
            add_statistic('ari', ari)
            add_statistic('ari_fg', ari_fg)

    print('evaluation finished')
    print(f'stats results: ')
    for k, v in stats.items():
        print(f'{k}: {v}')

