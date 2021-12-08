# Copyright @yucwang 2021
from os import listdir
from os.path import isfile, join, isdir, exists
import numpy as np
import torch 
import cv2 

class IOException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def write_matrix_to_exr(path, img):
    if not cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR)):
        raise IOException(f"Failed writing EXR: {path}")

def load_exr_to_matrix(path):
    return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB)

def load_exr_to_tensor(path):
    matrix = load_exr_to_matrix(path)
    tensor = torch.from_numpy(matrix).float().permute(2, 0, 1)
    return tensor

def write_tensor_to_exr(path, tensor, postprocess=False):
    if postprocess:
        tensor = postprocess_specular(tensor)
    matrix = tensor.permute(1, 2, 0).detach().cpu().numpy()
    write_matrix_to_exr(path, matrix)

def get_all_dirs(dir_path):
    dir_names = []
    for dir_name in listdir(dir_path):
        if isdir(join(dir_path, dir_name)):
            dir_names.append(dir_name)

    return dir_names

def preprocess_specular(specular):
    return torch.log(specular + 1.)

def preprocess_normal(normal):
    normal = torch.nan_to_num(normal)
    normal = (normal + 1.0) * 0.5
    normal = torch.clamp(normal, min=0.0, max=1.0)
    return normal

def postprocess_specular(specular):
    return torch.exp(specular) - 1.
