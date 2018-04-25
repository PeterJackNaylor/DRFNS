#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Data.CreateTFRecords import CreateTFRecord
from Data.ImageTransform import ListTransform
import numpy as np
from optparse import OptionParser
from utils import GetOptions


if __name__ == '__main__':

    options = GetOptions()

    OUTNAME = options.TFRecord
    PATH = options.path
    CROP = options.crop
    SIZE = options.size_train
    SPLIT = options.split
    var_elast = [1.3, 0.03, 0.15]
    var_he  = [0.01, 0.2]
    var_hsv = [0.2, 0.15]
    UNET = options.UNet
    SEED = options.seed
    N_EPOCH = options.epoch
    TYPE = options.type
    

    transform_list, transform_list_test = ListTransform(var_elast=var_elast,
                                                        var_hsv=var_hsv,
                                                        var_he=var_he) 
    if options.split == "train":
        TEST_PATIENT = ["testbreast", "testliver", "testkidney", "testprostate",
                        "bladder", "colorectal", "stomach", "validation"]
        TRANSFORM_LIST = transform_list_test
    elif options.split == "validation":
        options.split = "test"
        TEST_PATIENT = ["validation"]
        TRANSFORM_LIST = transform_list_test
        SIZE = options.size_test

    elif options.split == "test":
        TEST_PATIENT = ["testbreast", "testliver", "testkidney", "testprostate",
                        "bladder", "colorectal", "stomach"]
        TRANSFORM_LIST = transform_list_test
        SIZE = options.size_test
    elif options.split == "fulltrain":
        TEST_PATIENT = ["testbreast", "testliver", "testkidney", "testprostate",
                        "bladder", "colorectal", "stomach"]
        TRANSFORM_LIST = transform_list_test
    SIZE = (SIZE, SIZE)
    CreateTFRecord(OUTNAME, PATH, CROP, SIZE,
                   TRANSFORM_LIST, UNET, None, 
                   SEED, TEST_PATIENT, N_EPOCH,
                   TYPE=TYPE, SPLIT=SPLIT)
