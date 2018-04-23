#!/usr/bin/env python
# -*- coding: utf-8 -*-

from UNetBatchNorm import UNetBatchNorm
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score
from datetime import datetime
from DataReadDecode import read_and_decode
import os
from Net_Utils import EarlyStopper

def computeF1(P, GT, t):
    P = (P > t).astype('uint8').flatten()
    GT = (GT > t).astype('uint8').flatten()
    return f1_score(P, GT, pos_label=1)

class UNetDistance(UNetBatchNorm):
    def __init__(
        self,
        TF_RECORDS,
        LEARNING_RATE=0.01,
        K=0.96,
        BATCH_SIZE=10,
        IMAGE_SIZE=28,
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

        UNetBatchNorm.__init__(self, TF_RECORDS, LEARNING_RATE, K, 
            BATCH_SIZE, IMAGE_SIZE, 1, NUM_CHANNELS,
            NUM_TEST, STEPS, LRSTEP, DECAY_EMA, N_PRINT, LOG, 
            SEED, DEBUG, WEIGHT_DECAY, LOSS_FUNC, N_FEATURES,
             N_EPOCH, N_THREADS, MEAN_FILE, DROPOUT, EARLY_STOPPING)


    def init_training_graph(self):
        """
        Modified the evalution criteria so that it does the mean sqarred error
        instead of the cross-entropy loss on the DNN.
        """
        with tf.name_scope('Evaluation'):
            with tf.name_scope("logits/"):
                self.logits2 = tf.nn.conv2d(self.last, self.logits_weight, strides=[1,1,1,1], padding="VALID")
                self.logits = tf.nn.bias_add(self.logits2, self.logits_biases)
            self.predictions = self.logits
            with tf.name_scope('Loss'):

                self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.logits, self.train_labels_node))
                tf.summary.scalar("mean_squared_error", self.loss)
            self.predictions = tf.squeeze(self.predictions, [3])
            self.train_prediction = self.predictions

            self.test_prediction = self.predictions

        tf.global_variables_initializer().run()

        print('Computational graph initialised')

    def init_vars(self):
        """
        Different number of output channels.
        """
        self.is_training = tf.placeholder_with_default(True, shape=[])
        ####

        self.input_node = self.input_node_f()

        self.train_labels_node = self.label_node_f()
        n_features = self.N_FEATURES

        self.conv1_1weights = self.weight_xavier(3, self.NUM_CHANNELS, n_features, "conv1_1/")
        self.conv1_1biases = self.biases_const_f(0.1, n_features, "conv1_1/")

        self.conv1_2weights = self.weight_xavier(3, n_features, n_features, "conv1_2/")
        self.conv1_2biases = self.biases_const_f(0.1, n_features, "conv1_2/")

        self.conv1_3weights = self.weight_xavier(3, 2 * n_features, n_features, "conv1_3/")
        self.conv1_3biases = self.biases_const_f(0.1, n_features, "conv1_3/")

        self.conv1_4weights = self.weight_xavier(3, n_features, n_features, "conv1_4/")
        self.conv1_4biases = self.biases_const_f(0.1, n_features, "conv1_4/")



        self.conv2_1weights = self.weight_xavier(3, n_features, 2 * n_features, "conv2_1/")
        self.conv2_1biases = self.biases_const_f(0.1, 2 * n_features, "conv2_1/")

        self.conv2_2weights = self.weight_xavier(3, 2 * n_features, 2 * n_features, "conv2_2/")
        self.conv2_2biases = self.biases_const_f(0.1, 2 * n_features, "conv2_2/")

        self.conv2_3weights = self.weight_xavier(3, 4 * n_features, 2 * n_features, "conv2_3/")
        self.conv2_3biases = self.biases_const_f(0.1, 2 * n_features, "conv2_3/")

        self.conv2_4weights = self.weight_xavier(3, 2 * n_features, 2 * n_features, "conv2_4/")
        self.conv2_4biases = self.biases_const_f(0.1, 2 * n_features, "conv2_4/")



        self.conv3_1weights = self.weight_xavier(3, 2 * n_features, 4 * n_features, "conv3_1/")
        self.conv3_1biases = self.biases_const_f(0.1, 4 * n_features, "conv3_1/")

        self.conv3_2weights = self.weight_xavier(3, 4 * n_features, 4 * n_features, "conv3_2/")
        self.conv3_2biases = self.biases_const_f(0.1, 4 * n_features, "conv3_2/")

        self.conv3_3weights = self.weight_xavier(3, 8 * n_features, 4 * n_features, "conv3_3/")
        self.conv3_3biases = self.biases_const_f(0.1, 4 * n_features, "conv3_3/")

        self.conv3_4weights = self.weight_xavier(3, 4 * n_features, 4 * n_features, "conv3_4/")
        self.conv3_4biases = self.biases_const_f(0.1, 4 * n_features, "conv3_4/")



        self.conv4_1weights = self.weight_xavier(3, 4 * n_features, 8 * n_features, "conv4_1/")
        self.conv4_1biases = self.biases_const_f(0.1, 8 * n_features, "conv4_1/")

        self.conv4_2weights = self.weight_xavier(3, 8 * n_features, 8 * n_features, "conv4_2/")
        self.conv4_2biases = self.biases_const_f(0.1, 8 * n_features, "conv4_2/")

        self.conv4_3weights = self.weight_xavier(3, 16 * n_features, 8 * n_features, "conv4_3/")
        self.conv4_3biases = self.biases_const_f(0.1, 8 * n_features, "conv4_3/")

        self.conv4_4weights = self.weight_xavier(3, 8 * n_features, 8 * n_features, "conv4_4/")
        self.conv4_4biases = self.biases_const_f(0.1, 8 * n_features, "conv4_4/")



        self.conv5_1weights = self.weight_xavier(3, 8 * n_features, 16 * n_features, "conv5_1/")
        self.conv5_1biases = self.biases_const_f(0.1, 16 * n_features, "conv5_1/")

        self.conv5_2weights = self.weight_xavier(3, 16 * n_features, 16 * n_features, "conv5_2/")
        self.conv5_2biases = self.biases_const_f(0.1, 16 * n_features, "conv5_2/")




        self.tconv5_4weights = self.weight_xavier(2, 8 * n_features, 16 * n_features, "tconv5_4/")
        self.tconv5_4biases = self.biases_const_f(0.1, 8 * n_features, "tconv5_4/")

        self.tconv4_3weights = self.weight_xavier(2, 4 * n_features, 8 * n_features, "tconv4_3/")
        self.tconv4_3biases = self.biases_const_f(0.1, 4 * n_features, "tconv4_3/")

        self.tconv3_2weights = self.weight_xavier(2, 2 * n_features, 4 * n_features, "tconv3_2/")
        self.tconv3_2biases = self.biases_const_f(0.1, 2 * n_features, "tconv3_2/")

        self.tconv2_1weights = self.weight_xavier(2, n_features, 2 * n_features, "tconv2_1/")
        self.tconv2_1biases = self.biases_const_f(0.1, n_features, "tconv2_1/")



        self.logits_weight = self.weight_xavier(1, n_features, 1, "logits/")
        self.logits_biases = self.biases_const_f(0.1, 1, "logits/")

        self.keep_prob = tf.Variable(self.DROPOUT, name="dropout_prob")


    def init_model_architecture(self):
        """
        Graph structure.
        """
        self.conv1_1 = self.conv_layer_f(self.input_node, self.conv1_1weights, "conv1_1/")
        self.relu1_1 = self.relu_layer_f(self.conv1_1, self.conv1_1biases, "conv1_1/")

        self.conv1_2 = self.conv_layer_f(self.relu1_1, self.conv1_2weights, "conv1_2/")
        self.relu1_2 = self.relu_layer_f(self.conv1_2, self.conv1_2biases, "conv1_2/")


        self.pool1_2 = self.max_pool(self.relu1_2, name="pool1_2")


        self.conv2_1 = self.conv_layer_f(self.pool1_2, self.conv2_1weights, "conv2_1/")
        self.relu2_1 = self.relu_layer_f(self.conv2_1, self.conv2_1biases, "conv2_1/")

        self.conv2_2 = self.conv_layer_f(self.relu2_1, self.conv2_2weights, "conv2_2/")
        self.relu2_2 = self.relu_layer_f(self.conv2_2, self.conv2_2biases, "conv2_2/")        


        self.pool2_3 = self.max_pool(self.relu2_2, name="pool2_3")


        self.conv3_1 = self.conv_layer_f(self.pool2_3, self.conv3_1weights, "conv3_1/")
        self.relu3_1 = self.relu_layer_f(self.conv3_1, self.conv3_1biases, "conv3_1/")

        self.conv3_2 = self.conv_layer_f(self.relu3_1, self.conv3_2weights, "conv3_2/")
        self.relu3_2 = self.relu_layer_f(self.conv3_2, self.conv3_2biases, "conv3_2/")     


        self.pool3_4 = self.max_pool(self.relu3_2, name="pool3_4")


        self.conv4_1 = self.conv_layer_f(self.pool3_4, self.conv4_1weights, "conv4_1/")
        self.relu4_1 = self.relu_layer_f(self.conv4_1, self.conv4_1biases, "conv4_1/")

        self.conv4_2 = self.conv_layer_f(self.relu4_1, self.conv4_2weights, "conv4_2/")
        self.relu4_2 = self.relu_layer_f(self.conv4_2, self.conv4_2biases, "conv4_2/")


        self.pool4_5 = self.max_pool(self.relu4_2, name="pool4_5")


        self.conv5_1 = self.conv_layer_f(self.pool4_5, self.conv5_1weights, "conv5_1/")
        self.relu5_1 = self.relu_layer_f(self.conv5_1, self.conv5_1biases, "conv5_1/")

        self.conv5_2 = self.conv_layer_f(self.relu5_1, self.conv5_2weights, "conv5_2/")
        self.relu5_2 = self.relu_layer_f(self.conv5_2, self.conv5_2biases, "conv5_2/")



        self.tconv5_4 = self.transposeconv_layer_f(self.relu5_2, self.tconv5_4weights, "tconv5_4/")
        self.trelu5_4 = self.relu_layer_f(self.tconv5_4, self.tconv5_4biases, "tconv5_4/")
        self.bridge4 = self.CropAndMerge(self.relu4_2, self.trelu5_4, "bridge4")



        self.conv4_3 = self.conv_layer_f(self.bridge4, self.conv4_3weights, "conv4_3/")
        self.relu4_3 = self.relu_layer_f(self.conv4_3, self.conv4_3biases, "conv4_3/")

        self.conv4_4 = self.conv_layer_f(self.relu4_3, self.conv4_4weights, "conv4_4/")
        self.relu4_4 = self.relu_layer_f(self.conv4_4, self.conv4_4biases, "conv4_4/")



        self.tconv4_3 = self.transposeconv_layer_f(self.relu4_4, self.tconv4_3weights, "tconv4_3/")
        self.trelu4_3 = self.relu_layer_f(self.tconv4_3, self.tconv4_3biases, "tconv4_3/")
        self.bridge3 = self.CropAndMerge(self.relu3_2, self.trelu4_3, "bridge3")



        self.conv3_3 = self.conv_layer_f(self.bridge3, self.conv3_3weights, "conv3_3/")
        self.relu3_3 = self.relu_layer_f(self.conv3_3, self.conv3_3biases, "conv3_3/")

        self.conv3_4 = self.conv_layer_f(self.relu3_3, self.conv3_4weights, "conv3_4/")
        self.relu3_4 = self.relu_layer_f(self.conv3_4, self.conv3_4biases, "conv3_4/")



        self.tconv3_2 = self.transposeconv_layer_f(self.relu3_4, self.tconv3_2weights, "tconv3_2/")
        self.trelu3_2 = self.relu_layer_f(self.tconv3_2, self.tconv3_2biases, "tconv3_2/")
        self.bridge2 = self.CropAndMerge(self.relu2_2, self.trelu3_2, "bridge2")



        self.conv2_3 = self.conv_layer_f(self.bridge2, self.conv2_3weights, "conv2_3/")
        self.relu2_3 = self.relu_layer_f(self.conv2_3, self.conv2_3biases, "conv2_3/")

        self.conv2_4 = self.conv_layer_f(self.relu2_3, self.conv2_4weights, "conv2_4/")
        self.relu2_4 = self.relu_layer_f(self.conv2_4, self.conv2_4biases, "conv2_4/")



        self.tconv2_1 = self.transposeconv_layer_f(self.relu2_4, self.tconv2_1weights, "tconv2_1/")
        self.trelu2_1 = self.relu_layer_f(self.tconv2_1, self.tconv2_1biases, "tconv2_1/")
        self.bridge1 = self.CropAndMerge(self.relu1_2, self.trelu2_1, "bridge1")



        self.conv1_3 = self.conv_layer_f(self.bridge1, self.conv1_3weights, "conv1_3/")
        self.relu1_3 = self.relu_layer_f(self.conv1_3, self.conv1_3biases, "conv1_3/")

        self.conv1_4 = self.conv_layer_f(self.relu1_3, self.conv1_4weights, "conv1_4/")
        self.relu1_4 = self.relu_layer_f(self.conv1_4, self.conv1_4biases, "conv1_4/")

        self.last = self.relu1_4

        print('Model architecture initialised')



    def error_rate(self, predictions, labels, iter):
        """
        Error rate on the training data. To be displayed.
        """
        error = mean_squared_error(labels.flatten(), predictions.flatten())

        return error 

    def Validation(self, DG_TEST, step):
        """
        How the model validates.
        """
        if DG_TEST is None:
            print "no validation"
        else:
            n_test = DG_TEST.length
            loss, F1= 0., 0.
            for i in range(n_test):
                Xval, Yval = DG_TEST.Batch(0, 1)
                #Yval = Yval / 255.
                feed_dict = {self.input_node: Xval,
                             self.train_labels_node: Yval,
                             self.is_training: False}
                l_tmp, pred, s = self.sess.run([self.loss, 
                                                self.predictions,
                                                self.merged_summary],
                                                feed_dict=feed_dict)
                loss += l_tmp
                F1 += computeF1(pred, Yval, 0.5)


            loss, F1 = np.array([loss, F1]) / n_test

            summary = tf.Summary()
            summary.value.add(tag="TestMan/Loss", simple_value=loss)
            summary.value.add(tag="TestMan/F1", simple_value=F1)
            self.summary_test_writer.add_summary(summary, step) 
            self.summary_test_writer.add_summary(s, step) 
            print('  Validation loss: %.1f, F1: %.2f' % (loss, F1))
            self.saver.save(self.sess, self.LOG + '/' + "model.ckpt", step)
            wgt_path = self.LOG + '/' + "model.ckpt-{}".format(step)
            return loss, F1, wgt_path
    def train(self, DGTest):
        """
        How the model trains.
        """
        track = "F1"
        output = os.path.join(self.LOG, "data_collector.csv")
        look_behind = self.early_stopping_max
        early_stop = EarlyStopper(track, output, maximum=look_behind)

        epoch = self.STEPS * self.BATCH_SIZE // self.N_EPOCH
        self.Saver()
        trainable_var = tf.trainable_variables()
        self.LearningRateSchedule(self.LEARNING_RATE, self.K, epoch)
        self.optimization(trainable_var)
        self.ExponentialMovingAverage(trainable_var, self.DECAY_EMA)
        init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer(), self.data_init)
        self.sess.run(init_op)
        self.regularize_model()

        self.Saver()
        
        

        self.summary_test_writer = tf.summary.FileWriter(self.LOG + '/test',
                                            graph=self.sess.graph)

        self.summary_writer = tf.summary.FileWriter(self.LOG + '/train', graph=self.sess.graph)
        self.merged_summary = tf.summary.merge_all()
        steps = self.STEPS

        print "self.global step", int(self.global_step.eval())
        begin = int(self.global_step.eval())
        print "begin", begin
        for step in range(begin, steps + begin):  
            # self.optimizer is replaced by self.training_op for the exponential moving decay
            _, l, lr, predictions, batch_labels, s = self.sess.run(
                        [self.training_op, self.loss, self.learning_rate,
                         self.train_prediction, self.train_labels_node,
                         self.merged_summary])

            if step % self.N_PRINT == 0:
                i = datetime.now()
                print i.strftime('%Y/%m/%d %H:%M:%S: \n ')
                self.summary_writer.add_summary(s, step)                
                print('  Step %d of %d' % (step, steps))
                print('  Learning rate: %.5f \n') % lr
                print('  Mini-batch loss: %.5f \n ') % l
                print('  Max value: %.5f \n ') % np.max(predictions)
                values_test = self.Validation(DGTest, step)
                names_test = ["Loss", "F1", "wgt_path"]
                if early_stop.DataCollectorStopper(values_test, names_test, step):
                    break
        early_stop.save()
