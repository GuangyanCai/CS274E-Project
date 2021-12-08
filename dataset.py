# Copyright @yucwang 2021

from os.path import join
from cv2 import AlignExposures
import numpy as np
import random
import torch
from torch.functional import norm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from utils import load_exr_to_tensor, get_all_dirs, preprocess_normal, preprocess_specular

# Kujiale Dataset
class KujialeDataset(Dataset):
    """ Kujiale Dataset """

    def __init__(self, root_dir, ref_dir, transform=None, phase="train"):
        self.root_dir = root_dir
        self.ref_dir = ref_dir
        self.transform = transform
        self.id_list = get_all_dirs(root_dir)
        self.phase = phase
        self.id_list.sort()
        random.seed(13)
        random.shuffle(self.id_list)

        dataset_size = len(self.id_list)
        self.train_set_size = int(dataset_size * 0.8)
        self.val_set_size = int(dataset_size * 0.1)
        self.test_set_size = dataset_size - self.train_set_size - self.val_set_size

    def __getitem__(self, index):
        if self.phase == "validate":
            index = index + self.train_set_size
        elif self.phase == "test":
            index = index + self.train_set_size + self.val_set_size

        image_id = self.id_list[index]
        noisy_image_path = join(self.root_dir, image_id, "color.exr")
        normal_image_path = join(self.root_dir, image_id, "normal.exr")
        depth_image_path = join(self.root_dir, image_id, "depth.exr")
        albedo_image_path = join(self.root_dir, image_id, "texture.exr")
        ref_image_path = join(self.ref_dir, f"{image_id}xr.exr")

        noisy_image = load_exr_to_tensor(noisy_image_path)
        normal_image = load_exr_to_tensor(normal_image_path)
        depth_image = load_exr_to_tensor(depth_image_path)
        albedo_image = load_exr_to_tensor(albedo_image_path)
        ref_image = load_exr_to_tensor(ref_image_path)

        noisy_image = preprocess_specular(noisy_image)
        ref_image = preprocess_specular(ref_image)
        normal_image = preprocess_normal(normal_image)
        
        auxiliary_image = torch.cat([albedo_image, normal_image, depth_image[0:1, :, :]], dim=0)

        if self.transform:
            noisy_image = self.transform(noisy_image)
            auxiliary_image = self.transform(auxiliary_image)
            ref_image = self.transform(ref_image)
            
        return noisy_image, auxiliary_image, ref_image

    def __len__(self):
        if self.phase == "train":
            return self.train_set_size
        elif self.phase == "validate":
            return self.val_set_size
        else:
            return self.test_set_size
