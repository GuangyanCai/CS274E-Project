# Copyright @yucwang 2021

from os.path import join
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from utils import load_exr_to_matrix, get_all_dirs

# Kujiale Dataset
class KujialeFeaturesDataset():
    """ Kujiale Features Dataset """

    def __init__(self, root_dir, ref_dir, transform=None):
        self.root_dir = root_dir
        self.ref_dir = ref_dir
        self.transform = transform
        self.id_list = get_all_dirs(root_dir)

    def __getitem__(self, index):
        image_id = self.id_list[index]
        noisy_image_path = join(join(self.root_dir, image_id), "color.exr")
        normal_image_path = join(join(self.root_dir, image_id), "normal.exr")
        depth_image_path = join(join(self.root_dir, image_id), "depth.exr")
        albedo_image_path = join(join(self.root_dir, image_id), "texture.exr")
        ref_image_path = join(self.ref_dir, image_id+".exr")

        noisy_image = load_exr_to_matrix(noisy_image_path)
        normal_image = load_exr_to_matrix(normal_image_path)
        depth_image = load_exr_to_matrix(depth_image_path)
        albedo_image = load_exr_to_matrix(albedo_image_path)
        ref_image = load_exr_to_matrix(ref_image_path)

        return {"noisy_image": noisy_image, "normal_image": normal_image, "depth_image": depth_image, "albedo_image": albedo_image, "ref_image": ref_image }

    def __len__(self):
        return len(self.id_list)
