#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from datetime import datetime
 
class ConvolutionalNeuralNetwork:
    """
    Generic object for create DNN models.
    This class instinciates all functions 
    needed for DNN operations.
    """
    def __init__(
        self,
        LEARNING_RATE=0.01,
        K=0.96,
        BATCH_SIZE=1,
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
        EARLY_STOPPING=10):

        self.LEARNING_RATE = LEARNING_RATE
        self.K = K
        self.BATCH_SIZE = BATCH_SIZE
        self.IMAGE_SIZE = IMAGE_SIZE
        self.NUM_LABELS = NUM_LABELS
        self.NUM_CHANNELS = NUM_CHANNELS
        self.N_FEATURES = N_FEATURES
        self.STEPS = STEPS
        self.N_PRINT = N_PRINT
        self.LRSTEP = LRSTEP
        self.DECAY_EMA = DECAY_EMA
        self.LOG = LOG
        self.SEED = SEED

        self.sess = tf.InteractiveSession()

        self.sess.as_default()
        
        self.var_to_reg = []
        self.var_to_sum = []

        self.init_vars()
        self.init_model_architecture()
        self.init_training_graph()
        self.early_stopping_max = EARLY_STOPPING
        self.Saver()
        self.DEBUG = DEBUG
        self.loss_func = LOSS_FUNC
        self.weight_decay = WEIGHT_DECAY

    def regularize_model(self):
        """
        Adds regularization to parameters of the model given LOSS_FUNC
        """
        if self.DEBUG:
            for var in self.var_to_sum + self.var_to_reg:
                self.add_to_summary(var)
            self.WritteSummaryImages()
        
        for var in self.var_to_reg:
            self.add_to_regularization(var)



    def add_to_summary(self, var):
        """
        Adds histogram for each parameter in var
        """
        if var is not None:
            tf.summary.histogram(var.op.name, var)
    
    def add_to_regularization(self, var):
        """
        Combines loss with regularization loss
        """
        if var is not None:
            self.loss = self.loss + self.weight_decay * self.loss_func(var)


    def add_activation_summary(self, var):
        """
        Add activation summary with information about sparsity
        """
        if var is not None:
            tf.summary.histogram(var.op.name + "/activation", var)
            tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


    def add_gradient_summary(self, grad, var):
        """
        Add gradiant summary to summary
        """
        if grad is not None:
            tf.summary.histogram(var.op.name + "/gradient", grad)


    def input_node_f(self):
        """
        Input node, called when initialising the network
        """
        return tf.placeholder(
               tf.float32,
               shape=(self.BATCH_SIZE, self.IMAGE_SIZE, self.IMAGE_SIZE, self.NUM_CHANNELS))

    def label_node_f(self):
        """
        Label node, called when initialising the network
        """
        return tf.placeholder(
               tf.float32,
               shape=(self.BATCH_SIZE, self.IMAGE_SIZE, self.IMAGE_SIZE, 1))

    def conv_layer_f(self, i_layer, w_var, strides, scope_name, padding="SAME"):
        """
        Defining convolution layer
        """
        with tf.name_scope(scope_name):
            return tf.nn.conv2d(i_layer, w_var, strides=strides, padding=padding)

    def relu_layer_f(self, i_layer, biases, scope_name):
        """
        Defining relu layer
        """
        with tf.name_scope(scope_name):
            act = tf.nn.relu(tf.nn.bias_add(i_layer, biases))
            self.var_to_sum.append(act)
            return act

    def weight_const_f(self, ks, inchannels, outchannels, stddev, scope_name, name="W", reg="True"):
        """
        Defining parameter to give to a convolution layer
        """
        with tf.name_scope(scope_name):
            K = tf.Variable(tf.truncated_normal([ks, ks, inchannels, outchannels],  # 5x5 filter, depth 32.
                              stddev=stddev,
                              seed=self.SEED))
            self.var_to_reg.append(K)
            self.var_to_sum.append(K)
            return K

    def weight_xavier(self, ks, inchannels, outchannels, scope_name, name="W"):
        """
        Initialises a convolution kernel for a convolution layer with Xavier initialising
        """
        xavier_std = np.sqrt( 1. / float(ks * ks * inchannels) )
        return self.weight_const_f(ks, inchannels, outchannels, xavier_std, scope_name, name=name)

    def biases_const_f(self, const, shape, scope_name, name="B"):
        """
        Initialises biais
        """
        with tf.name_scope(scope_name):
            b = tf.Variable(tf.constant(const, shape=[shape]), name=name)
            self.var_to_sum.append(b)
            return b

    def max_pool(self, i_layer, ksize=[1,2,2,1], strides=[1,2,2,1],
                 padding="SAME", name="MaxPool"):
        """
        Performs max pool operation
        """
        return tf.nn.max_pool(i_layer, ksize=ksize, strides=strides, 
                              padding=padding, name=name)

    def BatchNorm(self, Input, n_out, phase_train, scope='bn', decay=0.9, eps=1e-5):
        """
        Performs batch normalisation.
        Code taken from http://stackoverflow.com/a/34634291/2267819
        """
        with tf.name_scope(scope):
            init_beta = tf.constant(0.0, shape=[n_out])
            beta = tf.Variable(init_beta, name="beta")
            init_gamma = tf.random_normal([n_out], 1.0, 0.02)
            gamma = tf.Variable(init_gamma)
            batch_mean, batch_var = tf.nn.moments(Input, [0, 1, 2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=decay)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(Input, mean, var, beta, gamma, eps)
        return normed

    def DropOutLayer(self, Input, scope="DropOut"):
        """
        Performs drop out on the input layer
        """
        with tf.name_scope(scope):
            return tf.nn.dropout(Input, self.keep_prob)  ##keep prob has to be defined in init_var

    def init_vars(self):
        """
        Initialises variables for the graph
        """
        self.input_node = self.input_node_f()

        self.train_labels_node = self.label_node_f()

        self.conv1_weights = self.weight_xavier(5, self.NUM_CHANNELS, 8, "conv1/")
        self.conv1_biases = self.biases_const_f(0.1, 8, "conv1/")

        self.conv2_weights = self.weight_xavier(5, 8, 8, "conv2/")
        self.conv2_biases = self.biases_const_f(0.1, 8, "conv2/")

        self.conv3_weights = self.weight_xavier(5, 8, 8, "conv3/")
        self.conv3_biases = self.biases_const_f(0.1, 8, "conv3/")

        self.logits_weight = self.weight_xavier(1, 8, self.NUM_LABELS, "logits/")
        self.logits_biases = self.biases_const_f(0.1, self.NUM_LABELS, "logits/")

        self.keep_prob = tf.Variable(0.5, name="dropout_prob")

        print('Model variables initialised')

    def WritteSummaryImages(self):
        """
        Image summary to add to the summary
        """
        tf.summary.image("Input", self.input_node, max_outputs=4)
        tf.summary.image("Label", self.train_labels_node, max_outputs=4)
        tf.summary.image("Pred", tf.expand_dims(tf.cast(self.predictions, tf.float32), dim=3), max_outputs=4)

    def init_model_architecture(self):
        """
        Graph structure for the model
        """
        self.conv1 = self.conv_layer_f(self.input_node, self.conv1_weights,
                                       [1,1,1,1], "conv1/")
        self.relu1 = self.relu_layer_f(self.conv1, self.conv1_biases, "conv1/")


        self.conv2 = self.conv_layer_f(self.relu1, self.conv2_weights,
                                       [1,1,1,1], "conv2/")
        self.relu2 = self.relu_layer_f(self.conv2, self.conv2_biases, "conv2/")

        self.conv3 = self.conv_layer_f(self.relu2, self.conv3_weights,
                                       [1,1,1,1], "conv3/")
        self.relu3 = self.relu_layer_f(self.conv3, self.conv3_biases, "conv3/")

        self.last = self.relu3

        print('Model architecture initialised')

    def init_training_graph(self):
        """
        Graph optimization part, here we define the loss and how the model is evaluated
        """

        with tf.name_scope('Evaluation'):
            self.logits = self.conv_layer_f(self.last, self.logits_weight, strides=[1,1,1,1], scope_name="logits/")
            self.predictions = tf.argmax(self.logits, axis=3)
            
            with tf.name_scope('Loss'):
                self.loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                          labels=tf.squeeze(tf.cast(self.train_labels_node, tf.int32), squeeze_dims=[3]),
                                                                          name="entropy")))
                tf.summary.scalar("entropy", self.loss)

            with tf.name_scope('Accuracy'):

                LabelInt = tf.squeeze(tf.cast(self.train_labels_node, tf.int64), squeeze_dims=[3])
                CorrectPrediction = tf.equal(self.predictions, LabelInt)
                self.accuracy = tf.reduce_mean(tf.cast(CorrectPrediction, tf.float32))
                tf.summary.scalar("accuracy", self.accuracy)

            with tf.name_scope('Prediction'):

                self.TP = tf.count_nonzero(self.predictions * LabelInt)
                self.TN = tf.count_nonzero((self.predictions - 1) * (LabelInt - 1))
                self.FP = tf.count_nonzero(self.predictions * (LabelInt - 1))
                self.FN = tf.count_nonzero((self.predictions - 1) * LabelInt)

            with tf.name_scope('Precision'):

                self.precision = tf.divide(self.TP, tf.add(self.TP, self.FP))
                tf.summary.scalar('Precision', self.precision)

            with tf.name_scope('Recall'):

                self.recall = tf.divide(self.TP, tf.add(self.TP, self.FN))
                tf.summary.scalar('Recall', self.recall)

            with tf.name_scope('F1'):

                num = tf.multiply(self.precision, self.recall)
                dem = tf.add(self.precision, self.recall)
                self.F1 = tf.scalar_mul(2, tf.divide(num, dem))
                tf.summary.scalar('F1', self.F1)

            with tf.name_scope('MeanAccuracy'):
                
                Nprecision = tf.divide(self.TN, tf.add(self.TN, self.FN))
                self.MeanAcc = tf.divide(tf.add(self.precision, Nprecision) ,2)
                tf.summary.scalar('Performance', self.MeanAcc)
            #self.batch = tf.Variable(0, name = "batch_iterator")

            self.train_prediction = tf.nn.softmax(self.logits)

            self.test_prediction = tf.nn.softmax(self.logits)

        tf.global_variables_initializer().run()

        print('Computational graph initialised')


    def error_rate(self, predictions, labels, iter):
        """
        Operations to perform on the training prediction every N_PRINT iterations.
        These values are printed to screen.
        """
        predictions = np.argmax(predictions, 3)
        labels = labels[:,:,:,0]

        cm = confusion_matrix(labels.flatten(), predictions.flatten(), labels=[0, 1]).astype(np.float)
        b, x, y = predictions.shape
        total = b * x * y

        TP = cm[1, 1]
        TN = cm[0, 0]
        FN = cm[0, 1]
        FP = cm[1, 0]

        acc = (TP + TN) / (TP + TN + FN + FP) * 100
        precision = TP / (TP + FP)
        acc1 = np.mean([precision, TN / (TN + FN)]) * 100
        recall = TP / (TP + FN)
        
        F1 = 2 * precision * recall / (recall + precision)
        error = 100 - acc

        return error, acc, acc1, recall * 100, precision * 100, F1 * 100

    def optimization(self, var_list):
        """
        Defining the optimization method to solve the task
        """
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            grads = optimizer.compute_gradients(self.loss, var_list=var_list)
            if self.DEBUG:
                for grad, var in grads:
                    self.add_gradient_summary(grad, var)
            self.optimizer = optimizer.apply_gradients(grads, global_step=self.global_step)        

    def LearningRateSchedule(self, lr, k, epoch):
        """
        Defines the learning rate
        """
        with tf.name_scope('LearningRateSchedule'):
            self.global_step = tf.Variable(0., trainable=False)
            tf.add_to_collection('global_step', self.global_step)
            if self.LRSTEP == "epoch/2":

                decay_step = float(epoch) / (2 * self.BATCH_SIZE)
            
            elif "epoch" in self.LRSTEP:
                num = int(self.LRSTEP[:-5])
                decay_step = float(num) * float(epoch) / self.BATCH_SIZE
            else:
                decay_step = float(self.LRSTEP)
              
            self.learning_rate = tf.train.exponential_decay(
                                     lr,
                                     self.global_step,
                                     decay_step,
                                     k,
                                     staircase=True)  
            tf.summary.scalar("learning_rate", self.learning_rate)

    def Validation(self, DG_TEST, step):
        """
        How the models validates on the test set.
        """
        n_test = DG_TEST.length
        n_batch = int(np.ceil(float(n_test) / self.BATCH_SIZE)) 

        l, acc, F1, recall, precision, meanacc = 0., 0., 0., 0., 0., 0.

        for i in range(n_batch):
            Xval, Yval = DG_TEST.Batch(0, self.BATCH_SIZE)
            feed_dict = {self.input_node: Xval,
                         self.train_labels_node: Yval}
            l_tmp, acc_tmp, F1_tmp, recall_tmp, precision_tmp, meanacc_tmp, pred = self.sess.run([self.loss, self.accuracy, self.F1, self.recall, self.precision, self.MeanAcc, self.predictions], feed_dict=feed_dict)
            l += l_tmp
            acc += acc_tmp
            F1 += F1_tmp
            recall += recall_tmp
            precision += precision_tmp
            meanacc += meanacc_tmp

        l, acc, F1, recall, precision, meanacc = np.array([l, acc, F1, recall, precision, meanacc]) / n_batch


        summary = tf.Summary()
        summary.value.add(tag="Test/Accuracy", simple_value=acc)
        summary.value.add(tag="Test/Loss", simple_value=l)
        summary.value.add(tag="Test/F1", simple_value=F1)
        summary.value.add(tag="Test/Recall", simple_value=recall)
        summary.value.add(tag="Test/Precision", simple_value=precision)
        summary.value.add(tag="Test/Performance", simple_value=meanacc)
        self.summary_test_writer.add_summary(summary, step) 

        print('  Validation loss: %.1f' % l)
        print('       Accuracy: %1.f%% \n       acc1: %.1f%% \n       recall: %1.f%% \n       prec: %1.f%% \n       f1 : %1.f%% \n' % (acc * 100, meanacc * 100, recall * 100, precision * 100, F1 * 100))
        self.saver.save(self.sess, self.LOG + '/' + "model.ckpt", global_step=self.global_step)

    def Saver(self):
        """
        Defining the saver, it will load if possible.
        """
        print("Setting up Saver...")
        self.saver = tf.train.Saver(max_to_keep=(self.early_stopping_max + 1))
        ckpt = tf.train.get_checkpoint_state(self.LOG)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model restored...")

    def ExponentialMovingAverage(self, var_list, decay=0.9999):
        """
        Adding exponential moving average to increase performance.
        This aggregates parameters from different steps in order to have
        a more robust classifier.
        """
        with tf.name_scope('ExponentialMovingAverage'):

            ema = tf.train.ExponentialMovingAverage(decay=decay)
            maintain_averages_op = ema.apply(var_list)

            # Create an op that will update the moving averages after each training
            # step.  This is what we will use in place of the usual training op.
            with tf.control_dependencies([self.optimizer]):
                self.training_op = tf.group(maintain_averages_op)

    def train(self, DGTrain, DGTest, saver=True):
        """
        How the model should train.
        """

        epoch = DGTrain.length

        self.LearningRateSchedule(self.LEARNING_RATE, self.K, epoch)

        trainable_var = tf.trainable_variables()
        
        self.regularize_model()
        self.optimization(trainable_var)
        self.ExponentialMovingAverage(trainable_var, self.DECAY_EMA)

        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        self.summary_test_writer = tf.summary.FileWriter(self.LOG + '/test',
                                            graph=self.sess.graph)

        self.summary_writer = tf.summary.FileWriter(self.LOG + '/train', graph=self.sess.graph)
        merged_summary = tf.summary.merge_all()
        steps = self.STEPS


        for step in range(steps):
            batch_data, batch_labels = DGTrain.Batch(0, self.BATCH_SIZE)
            feed_dict = {self.input_node: batch_data,
                         self.train_labels_node: batch_labels}

            # self.optimizer is replaced by self.training_op for the exponential moving decay
            _, l, lr, predictions, s = self.sess.run(
                        [self.training_op, self.loss, self.learning_rate,
                         self.train_prediction, merged_summary],
                        feed_dict=feed_dict)

            if step % self.N_PRINT == 0:
                i = datetime.now()
                print i.strftime('%Y/%m/%d %H:%M:%S: \n ')
                self.summary_writer.add_summary(s, step)                
                error, acc, acc1, recall, prec, f1 = self.error_rate(predictions, batch_labels, step)
                print('  Step %d of %d' % (step, steps))
                print('  Learning rate: %.5f \n') % lr
                print('  Mini-batch loss: %.5f \n       Accuracy: %.1f%% \n       acc1: %.1f%% \n       recall: %1.f%% \n       prec: %1.f%% \n       f1 : %1.f%% \n' % 
                      (l, acc, acc1, recall, prec, f1))
                self.Validation(DGTest, step)
    def predict(self, tensor):
        feed_dict = {self.input_node: tensor,
                     self.is_training: False}
        pred = self.sess.run(self.predictions,
                            feed_dict=feed_dict)
        return pred
