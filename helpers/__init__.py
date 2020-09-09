import os

import torch
from tqdm import tqdm


def load_epoch(save_path, epoch):
    tqdm.write('loading from epoch.%04d.pth' % epoch)
    return torch.load(os.path.join(save_path, 'epoch.%04d.pth' % epoch),
                      map_location='cpu')
