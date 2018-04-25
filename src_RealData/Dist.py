#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import GetOptions, ComputeMetrics, CheckOrCreate
import tensorflow as tf
import numpy as np
import os
from glob import glob
from Data.ImageTransform import ListTransform
from Data.DataGenClass import DataGenMulti
from Nets.DataReadDecode import read_and_decode
from Nets.UNetDistance import UNetDistance
from skimage import img_as_ubyte


class Model(UNetDistance):
    def validation(self, p1, p2, steps):
        """
        How the model validates
        """
        loss, roc = 0., 0.
        acc, F1, recall = 0., 0., 0.
        precision, jac, AJI = 0., 0., 0.
        init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer(), self.data_init)
        self.sess.run(init_op)
        self.Saver()
        
        for step in range(steps):  
            feed_dict = {self.is_training: False} 
            l, prob, batch_labels = self.sess.run([self.loss, self.predictions,
                                                              self.train_labels_node], feed_dict=feed_dict)
            prob[prob > 255] = 255
            prob[prob < 0] = 0
            prob = prob.astype(int)
            batch_labels[batch_labels > 0] = 255
            loss += l
            out = ComputeMetrics(prob[0], batch_labels[0], p1, p2)
            acc += out[0]
            roc += out[1]
            jac += out[2]
            recall += out[3]
            precision += out[4]
            F1 += out[5]
            AJI += out[6]
        loss, acc, F1 = np.array([loss, acc, F1]) / steps
        recall, precision, roc = np.array([recall, precision, roc]) / steps
        jac, AJI = np.array([jac, AJI]) / steps
        return loss, acc, F1, recall, precision, roc, jac, AJI

    def test(self, DG_TEST, p1, p2, save_path):
        """
        How the model test
        """
        n_test = DG_TEST.length
        n_batch = int(np.ceil(float(n_test) / self.BATCH_SIZE)) 
        res = []

        for i in range(n_batch):
            Xval, Yval = DG_TEST.Batch(0, self.BATCH_SIZE)
            feed_dict = {self.input_node: Xval,
                         self.train_labels_node: Yval,
                         self.is_training: False}
            l, pred = self.sess.run([self.loss, self.predictions],
                                    feed_dict=feed_dict)
            #pred[pred > 255] = 255
            #pred[pred < 0] = 0
            #pred = pred / 255.
            #pred = img_as_ubyte(pred)
            rgb = (Xval[0,92:-92,92:-92] + np.load(self.MEAN_FILE)).astype(np.uint8)
            Yval[ Yval > 0 ] = 255
            out = ComputeMetrics(pred[0,:,:], Yval[0,:,:,0], p1, p2, rgb=rgb, save_path=save_path, ind=i, uint8=False)
            out = [l] + list(out)
            res.append(out)
        return res



if __name__== "__main__":

    transform_list, transform_list_test = ListTransform(n_elastic=0)
    options = GetOptions()

    SPLIT = options.split
    if SPLIT == "fulltrain":
        SPLIT = "train"
    ## Model parameters
    TFRecord = options.TFRecord
    LEARNING_RATE = options.lr
    BATCH_SIZE = options.bs
    SIZE = (options.size_train, options.size_train)
    if options.size_test is not None:
        SIZE = (options.size_test, options.size_test)
    N_ITER_MAX = 0 ## defined later
    LRSTEP = "10epoch"
    if SPLIT == "train":
        samples_per_epoch = len([0 for record in tf.python_io.tf_record_iterator(options.TFRecord)] )
        N_TRAIN_SAVE = samples_per_epoch // BATCH_SIZE
    else:
        N_TRAIN_SAVE = 100
    LOG = options.log
    WEIGHT_DECAY = options.weight_decay
    N_FEATURES = options.n_features
    N_EPOCH = options.epoch
    N_THREADS = options.THREADS
    MEAN_FILE = options.mean_file 
    DROPOUT = options.dropout

    ## Datagen parameters
    PATH = options.path

    TEST_PATIENT = ["testbreast", "testliver", "testkidney", "testprostate",
                        "bladder", "colorectal", "stomach", "validation"]
    DG_TRAIN = DataGenMulti(PATH, split='train', crop = 16, size=SIZE,
                       transforms=transform_list_test, UNet=True, mean_file=None, num=TEST_PATIENT)
    TEST_PATIENT = ["validation"]
    DG_TEST  = DataGenMulti(PATH, split="test", crop = 1, size=(996, 996), 
                       transforms=transform_list_test, UNet=True, mean_file=MEAN_FILE, num=TEST_PATIENT)
    if SPLIT == "train":
        N_ITER_MAX = N_EPOCH * DG_TRAIN.length // BATCH_SIZE
    elif SPLIT == "validation":
        N_ITER_MAX = N_EPOCH * DG_TEST.length // BATCH_SIZE
#    elif SPLIT == "test":
#        LOG = glob(os.path.join(LOG, '*'))[0]
    model = Model(TFRecord,            LEARNING_RATE=LEARNING_RATE,
                                       BATCH_SIZE=BATCH_SIZE,
                                       IMAGE_SIZE=SIZE,
                                       NUM_CHANNELS=3,
                                       STEPS=N_ITER_MAX,
                                       LRSTEP=LRSTEP,
                                       N_PRINT=N_TRAIN_SAVE,
                                       LOG=LOG,
                                       SEED=42,
                                       WEIGHT_DECAY=WEIGHT_DECAY,
                                       N_FEATURES=N_FEATURES,
                                       N_EPOCH=N_EPOCH,
                                       N_THREADS=N_THREADS,
                                       MEAN_FILE=MEAN_FILE,
                                       DROPOUT=DROPOUT,
                                       EARLY_STOPPING=30)
    
    if SPLIT == "train":
        DG_TEST  = DataGenMulti(PATH, split="test", crop = 16, size=SIZE,
                           transforms=transform_list_test, UNet=True, mean_file=MEAN_FILE, num=TEST_PATIENT)
        model.train(DG_TEST)
    elif SPLIT == "validation":
        p1 = options.p1
        p2 = options.p2
        file_name = options.output
        f = open(file_name, 'w')
        outs = model.validation(p1, p2, N_ITER_MAX)
        outs = [LOG] + list(outs) + [p1, p2]
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


        for organ in TEST_PATIENT:
            DG_TEST  = DataGenMulti(PATH, split="test", crop = 1, size=(996, 996),num=[organ],
                           transforms=transform_list_test, UNet=True, mean_file=MEAN_FILE)
            save_organ = os.path.join(options.save_path, organ)
            CheckOrCreate(save_organ)
            outs = model.test(DG_TEST, options.p1, options.p2, save_organ)
            for i in range(len(outs)):
                small_o = outs[i]
                small_o = [i, organ] + small_o + [options.p1, options.p2]
                f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(*small_o))
        f.close()

