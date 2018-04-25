#!/usr/bin/env python
# -*- coding: utf-8 -*-

from FCN_Object import FCN8
from utils import GetOptions, ComputeMetrics, CheckOrCreate
import tensorflow as tf
import numpy as np
import os
from Data.ImageTransform import ListTransform
from Nets.DataReadDecode import read_and_decode
from Data.DataGenClass import DataGenMulti

if __name__== "__main__":

    options = GetOptions()

    SPLIT = options.split
    if SPLIT == "fulltrain":
        SPLIT = "train"

    ## Model parameters
    TFRecord = options.TFRecord
    LEARNING_RATE = options.lr
    SIZE = (options.size_train, options.size_train)
    if options.size_test is not None:
        SIZE = (options.size_test, options.size_test)
    N_ITER_MAX = options.iters
    N_TRAIN_SAVE = 1000
    MEAN_FILE = options.mean_file 
    save_dir = options.log
    checkpoint = options.restore

    model = FCN8(checkpoint, save_dir, TFRecord, SIZE[0],
                 2, 1000, options.split)
    if SPLIT == "train":
        model.train8(N_ITER_MAX, LEARNING_RATE)
    elif SPLIT == "validation":
        p1 = options.p1
        LOG = options.log

        file_name = options.output
        f = open(file_name, 'w')

        checkpoint = os.path.join(checkpoint, checkpoint) 
        outs = model.validation8(N_ITER_MAX, checkpoint, p1)
        outs = [LOG] + list(outs) + [p1, 0.5]
        NAMES = ["ID", "Loss", "Acc", "F1", "Recall", "Precision", "ROC", "Jaccard", "AJI", "p1", "p2"]
        f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(*NAMES))

        f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(*outs))
    elif SPLIT == "test":
        
        TEST_PATIENT = ["testbreast", "testliver", "testkidney", "testprostate",
                        "bladder", "colorectal", "stomach"]

        file_name = options.output
        f = open(file_name, 'w')
        NAMES = ["NUMBER", "ORGAN", "Loss", "Acc", "ROC", "Jaccard", "Recall", "Precision", "F1", "AJI", "p1", "p2"]
        f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(*NAMES))
        transform_list, transform_list_test = ListTransform()
        PATH = options.path
        for organ in TEST_PATIENT:
            DG_TEST  = DataGenMulti(PATH, split="test", crop = 1, size=(1000, 1000),num=[organ],
                           transforms=transform_list_test, UNet=False, mean_file=None)
            save_organ = os.path.join(options.save_path, organ)
            CheckOrCreate(save_organ)
            outs = model.test8(DG_TEST, 2, options.p1, 0.5, save_organ)
            for i in range(len(outs)):
                small_o = outs[i]
                small_o = [i, organ] + small_o + [options.p1, 0.5]
                f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(*small_o))
        f.close()
