# Copyright @yucwang 2021

import numpy as np
from os import listdir
from os.path import isfile, join, isdir, exists
import OpenEXR
import Imath

class IOException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def write_matrix_to_exr(path, img):
    try:
        img = np.squeeze(img)
        sz = img.shape
        header = OpenEXR.Header(sz[1], sz[0])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        header['channels'] = dict([(c, half_chan) for c in "RGB"])
        out = OpenEXR.OutputFile(path, header)
        R = (img[:,:,0]).astype(np.float16).tobytes()
        G = (img[:,:,1]).astype(np.float16).tobytes()
        B = (img[:,:,2]).astype(np.float16).tobytes()
        out.writePixels({'R' : R, 'G' : G, 'B' : B})
        out.close()
    except Exception as e:
        raise IOException("Failed writing EXR: %s"%e)

def load_exr_to_matrix(path,channel=3):
    image = OpenEXR.InputFile(path)
    dataWindow = image.header()['dataWindow']
    size = (dataWindow.max.x - dataWindow.min.x + 1, dataWindow.max.y - dataWindow.min.y + 1)
    HALF = Imath.PixelType(Imath.PixelType.HALF)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    if channel == 3:
        data = np.array([np.fromstring(image.channel(c, FLOAT), dtype=np.float32) for c in 'BGR'])
    elif channel == 4:
        data = np.array([np.fromstring(image.channel(c, FLOAT), dtype=np.float32) for c in 'BGRA'])
    data = np.moveaxis(data, 0, -1)
    data = data.reshape(size[1], size[0],channel)
    return data

def get_all_dirs(dir_path):
    dir_names = []
    for dir_name in listdir(dir_path):
        if isdir(join(dir_path, dir_name)):
            dir_names.append(dir_name)

    return dir_names


exr_file = load_exr_to_matrix("/home/yucwang/Downloads/KJL_features/L3D101IYE6QBAUPFR7O7XE3P3WE888/color.exr")
write_matrix_to_exr("/home/yucwang/Desktop/test.exr", exr_file)
# print(exr_file.shape)

# file_lists = get_all_dirs("/home/yucwang/Downloads/KJL_features/")
# print(file_lists)
