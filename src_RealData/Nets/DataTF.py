#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ObjectOriented import ConvolutionalNeuralNetwork
import os
import tensorflow as tf
from datetime import datetime
import numpy as np
from DataReadDecode import read_and_decode


class DataReader(ConvolutionalNeuralNetwork):
    """
    Class for TF modules with record read for inputing the images
    and labels to the DNN.
    """
    def __init__(
        self,
        TF_RECORDS,
        LEARNING_RATE=0.01,
        K=0.96,
        BATCH_SIZE=10,
        IMAGE_SIZE=28,
        NUM_LABELS=10,
        NUM_CHANNELS=1,
        NUM_TEST=10000,
        STEPS=2000,
        LRSTEP=200,
        DECAY_EMA=0.9999,
        N_PRINT = 100,
        LOG="/tmp/net",
        SEED=42,
        DEBUG=True,
        WEIGHT_DECAY=0.00005,
        LOSS_FUNC=tf.nn.l2_loss,
        N_FEATURES=16,
        N_EPOCH=1,
        N_THREADS=1,
        MEAN_FILE=None,
        DROPOUT=0.5,
        EARLY_STOPPING=10):

        self.N_EPOCH = N_EPOCH
        self.N_THREADS = N_THREADS
        self.DROPOUT = DROPOUT
        self.MEAN_FILE = MEAN_FILE
        if MEAN_FILE is not None:
            MEAN_ARRAY = tf.constant(np.load(MEAN_FILE), dtype=tf.float32) # (3)
            self.MEAN_ARRAY = tf.reshape(MEAN_ARRAY, [1, 1, NUM_CHANNELS])
            self.SUB_MEAN = True
        else:
            self.SUB_MEAN = False
        self.TF_RECORDS = TF_RECORDS
        self.IMAGE_SIZE = IMAGE_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.init_queue(TF_RECORDS)
        ConvolutionalNeuralNetwork.__init__(self, LEARNING_RATE, K, 
            BATCH_SIZE, IMAGE_SIZE, NUM_LABELS, NUM_CHANNELS, 
            NUM_TEST, STEPS, LRSTEP, DECAY_EMA, N_PRINT, LOG, 
            SEED, DEBUG, WEIGHT_DECAY, LOSS_FUNC, N_FEATURES,
            EARLY_STOPPING)

    def init_queue(self, tfrecords_filename):
        """
        New queues for coordinator
        """
        with tf.device('/cpu:0'):
            self.data_init, self.image, self.annotation = read_and_decode(tfrecords_filename, 
                                                          self.IMAGE_SIZE[0], 
                                                          self.IMAGE_SIZE[1],
                                                          self.BATCH_SIZE,
                                                          self.N_THREADS)
        print("Queue initialized")
    def input_node_f(self):
        """
        The input node can now come from the record or can be inputed
        via a feed dict (for testing for example)
        """
        if self.SUB_MEAN:
            self.images_queue = self.image - self.MEAN_ARRAY
        else:
            self.images_queue = self.image
        self.image_PH = tf.placeholder_with_default(self.images_queue, shape=[None,
                                                                              None, 
                                                                              None,
                                                                              3])
        return self.image_PH
    def label_node_f(self):
        """
        Same for input node f
        """
        self.labels_queue = self.annotation
        self.labels_PH = tf.placeholder_with_default(self.labels_queue, shape=[None,
                                                                          None, 
                                                                          None,
                                                                          1])

        return self.labels_PH
    def Validation(self, DG_TEST, step):
        """
        How the models validates it self with respect to the test set
        """
        if DG_TEST is None:
            print "no validation"
        else:
            n_test = DG_TEST.length
            n_batch = int(np.ceil(float(n_test) / self.BATCH_SIZE)) 

            l, acc, F1, recall, precision, meanacc = 0., 0., 0., 0., 0., 0.

            for i in range(n_batch):
                Xval, Yval = DG_TEST.Batch(0, self.BATCH_SIZE)
                feed_dict = {self.input_node: Xval,
                             self.train_labels_node: Yval}
                l_tmp, acc_tmp, F1_tmp, recall_tmp, precision_tmp, meanacc_tmp, pred, s = self.sess.run([self.loss, 
                                                                                            self.accuracy, self.F1,
                                                                                            self.recall, self.precision,
                                                                                            self.MeanAcc, self.predictions,
                                                                                            self.merged_summary], feed_dict=feed_dict)
                l += l_tmp
                acc += acc_tmp
                F1 += F1_tmp
                recall += recall_tmp
                precision += precision_tmp
                meanacc += meanacc_tmp

            l, acc, F1, recall, precision, meanacc = np.array([l, acc, F1, recall, precision, meanacc]) / n_batch

            summary = tf.Summary()
            summary.value.add(tag="TestMan/Accuracy", simple_value=acc)
            summary.value.add(tag="TestMan/Loss", simple_value=l)
            summary.value.add(tag="TestMan/F1", simple_value=F1)
            summary.value.add(tag="TestMan/Recall", simple_value=recall)
            summary.value.add(tag="TestMan/Precision", simple_value=precision)
            summary.value.add(tag="TestMan/Performance", simple_value=meanacc)
            self.summary_test_writer.add_summary(summary, step) 
            self.summary_test_writer.add_summary(s, step) 
            print('  Validation loss: %.1f' % l)
            print('       Accuracy: %1.f%% \n       acc1: %.1f%% \n       recall: %1.f%% \n       prec: %1.f%% \n       f1 : %1.f%% \n' % (acc * 100, meanacc * 100, recall * 100, precision * 100, F1 * 100))
            self.saver.save(self.sess, self.LOG + '/' + "model.ckpt", step)

    def train(self, DG_TEST=None):
        """
        How the model trains
        """

        epoch = self.STEPS * self.BATCH_SIZE // self.N_EPOCH

        self.LearningRateSchedule(self.LEARNING_RATE, self.K, epoch)

        trainable_var = tf.trainable_variables()
        
        self.regularize_model()
        self.optimization(trainable_var)
        self.ExponentialMovingAverage(trainable_var, self.DECAY_EMA)

        self.summary_test_writer = tf.summary.FileWriter(self.LOG + '/test',
                                            graph=self.sess.graph)

        self.summary_writer = tf.summary.FileWriter(self.LOG + '/train', graph=self.sess.graph)
        self.merged_summary = tf.summary.merge_all()
        steps = self.STEPS

        init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer(), self.data_init)
        self.sess.run(init_op)

        for step in range(steps):      
            # self.optimizer is replaced by self.training_op for the exponential moving decay
            _, l, lr, predictions, batch_labels, s = self.sess.run(
                        [self.training_op, self.loss, self.learning_rate,
                         self.train_prediction, self.train_labels_node,
                         self.merged_summary])

            if step % self.N_PRINT == 0:
                i = datetime.now()
                print i.strftime('%Y/%m/%d %H:%M:%S: \n ')
                self.summary_writer.add_summary(s, step)                
                error, acc, acc1, recall, prec, f1 = self.error_rate(predictions, batch_labels, step)
                print('  Step %d of %d' % (step, steps))
                print('  Learning rate: %.5f \n') % lr
                print('  Mini-batch loss: %.5f \n       Accuracy: %.1f%% \n       acc1: %.1f%% \n       recall: %1.f%% \n       prec: %1.f%% \n       f1 : %1.f%% \n' % 
                     (l, acc, acc1, recall, prec, f1))
                self.Validation(DG_TEST, step)
