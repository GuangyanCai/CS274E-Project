# Copyright @yucwang 2021

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Kujiale Dataset
class KujialeFeaturesDataset():
    """ Kujiale Features Dataset """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
