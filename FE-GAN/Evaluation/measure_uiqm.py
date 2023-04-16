# -*- encoding:utf-8 -*-
# @Author: jhan
# @FileName: measure_uiqm.py
# @Date: 2022/11/7 10:30


"""
#   - Underwater Image Quality Measure (UIQM)
#   - UIQM = c1UICM + c2UISM + c3UIConM, here c1 = 0.0282, c2 = 0.2953, c3 = 3.5753.
"""


# python libs
import numpy as np
from PIL import Image, ImageOps
from glob import glob
from os.path import join
from ntpath import basename
# local libs
from uqim_utils import getUIQM
from uqim_utils import getUICM
from uqim_utils import getUISM
from uqim_utils import getUIConM


def measure_UIQMs(dir_name, im_res=(256, 256)):
    paths = sorted(glob(join(dir_name, "*.*")))
    uiqms = []
    for img_path in paths:
        im = Image.open(img_path).resize(im_res)
        uiqm = getUIQM(np.array(im))
        uiqms.append(uiqm)
    return np.array(uiqms)


def measure_UICMs(dir_name, im_res=(256, 256)):
    paths = sorted(glob(join(dir_name, "*.*")))
    uicms = []
    for img_path in paths:
        im = Image.open(img_path).resize(im_res)
        uicm = getUICM(np.array(im))
        uicms.append(uicm)
    return np.array(uicms)

def measure_UISMs(dir_name, im_res=(256, 256)):
    paths = sorted(glob(join(dir_name, "*.*")))
    uisms = []
    for img_path in paths:
        im = Image.open(img_path).resize(im_res)
        uism = getUISM(np.array(im))
        uisms.append(uism)
    return np.array(uisms)

def measure_UIConMs(dir_name, im_res=(256, 256)):
    paths = sorted(glob(join(dir_name, "*.*")))
    uiconms = []
    for img_path in paths:
        im = Image.open(img_path).resize(im_res)
        uiconm = getUIConM(np.array(im))
        uiconms.append(uiconm)
    return np.array(uiconms)


if __name__ == '__main__':
    # distorted input im paths
    inp_dir = "./PyTorch/Data/EUVP/Test/Inp/"

    # UIQMs of the distorted input images
    inp_uqims = measure_UIQMs(inp_dir)
    print("Input UIQMs >> Mean: {0} std: {1}".format(np.mean(inp_uqims), np.std(inp_uqims)))

    # generated output im paths
    gen_dir = "./PyTorch/results/"

    # UICMs of the enhanceded output images
    gen_uqims = measure_UICMs(gen_dir)
    print("Enhanced UICMs >> Mean: {0} std: {1}".format(np.mean(gen_uqims), np.std(gen_uqims)))

    # UISMs of the enhanceded output images
    gen_uqims = measure_UISMs(gen_dir)
    print("Enhanced UISMs >> Mean: {0} std: {1}".format(np.mean(gen_uqims), np.std(gen_uqims)))

    # UIConMs of the enhanceded output images
    gen_uqims = measure_UIConMs(gen_dir)
    print("Enhanced UIConMs >> Mean: {0} std: {1}".format(np.mean(gen_uqims), np.std(gen_uqims)))

    # UIQMs of the enhanceded output images
    gen_uqims = measure_UIQMs(gen_dir)
    print("Enhanced UIQMs >> Mean: {0} std: {1}".format(np.mean(gen_uqims), np.std(gen_uqims)))

