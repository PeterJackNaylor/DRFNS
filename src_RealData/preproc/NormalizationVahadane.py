#!/usr/bin/env python
# -*- coding: utf-8 -*-

## To use this code, you will need python 3 (> 3.5)
# Clone this repo: https://github.com/Peter554/StainTools
# and install the python package SPAMS: https://anaconda.org/conda-forge/python-spams


### This code is to only be used to normalized the data before feeding it into the nextflow pipeline



from optparse import OptionParser
from glob import glob
from tqdm import *
import os
from skimage.io import imsave
from shutil import copyfile


## Beware the following utils folder is from the StainTools repo code.
## Due to this naming problem, we just redefine the function here.
def CheckOrCreate(path):
    """
    If path exists, does nothing otherwise it creates it.
    """
    if not os.path.isdir(path):
        os.makedirs(path)

## Packages
from utils import visual_utils as vu
from normalization.reinhard import ReinhardNormalizer
from normalization.macenko import MacenkoNormalizer
from normalization.vahadane import VahadaneNormalizer

def OptionsMain():
    parser = OptionParser()
    parser.add_option('--input', dest="input", type="str")
    parser.add_option('--output', dest="output", type="str")
    parser.add_option('--target', dest="target", type="str")
    parser.add_option('--normalization', dest="normalization", type="str")
    (options, args) = parser.parse_args()
    return options

def CollectFiles(input):
    return glob(os.path.join(input, "Slide_*", "Slide_*.png"))

def GT_path(rgb_path):
    return rgb_path.replace('Slide', 'GT')

def SavePath(path, input, output):
    out_name = path.replace(input, output)
    CheckOrCreate(os.path.dirname(out_name))
    return out_name

def FindTarget(target, list):
    for element in list:
        if target in element:
            return element
    else:
        assert 0 == 1, "Target not in input path"

def PrepNormalizer(normalization, targetPath):
    if "Reinhard" in normalization:
        n = ReinhardNormalizer()
    elif "Macenko" in normalization:
        n = MacenkoNormalizer()
    elif "Vahadane":
        n = VahadaneNormalizer()
    else:
        print("No knowned normalization given..")
    targetImg = vu.read_image(targetPath)
    n.fit(targetImg)
    return n

def TransformGenerator(img_paths, normalizer, target):
    for f in img_paths:
        img = vu.read_image(f)
        if f == target: 
            new_img = img
        else:
            new_img = normalizer.transform(img)
        yield new_img, f

if __name__ == '__main__':

    options = OptionsMain()
    files = CollectFiles(options.input)
    target_file = FindTarget(options.target, files)
    n = PrepNormalizer(options.normalization, target_file)

    for t_img, name in tqdm(TransformGenerator(files, n, target_file)):
        gt_name = GT_path(name)
        out_name = SavePath(name, options.input, options.output)
        out_gt_name = SavePath(gt_name, options.input, options.output)
        imsave(out_name, t_img)
        copyfile(gt_name, out_gt_name)

