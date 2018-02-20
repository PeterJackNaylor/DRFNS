#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Nets.UNetBatchNorm import UNetBatchNorm
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error
from datetime import datetime
from optparse import OptionParser
from Data.ImageTransform import ListTransform
from Data.DataGenClass import DataGen3, DataGenMulti, DataGen3reduce
from Nets.DataReadDecode import read_and_decode
import pdb
from skimage.io import imsave
from os.path import join
import os
from utils import CheckOrCreate

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
        DROPOUT=0.5):

        self.LEARNING_RATE = LEARNING_RATE
        self.K = K
        self.BATCH_SIZE = BATCH_SIZE
        self.IMAGE_SIZE = IMAGE_SIZE
        self.NUM_CHANNELS = NUM_CHANNELS
        self.N_FEATURES = N_FEATURES
#        self.NUM_TEST = NUM_TEST
        self.STEPS = STEPS
        self.N_PRINT = N_PRINT
        self.LRSTEP = LRSTEP
        self.DECAY_EMA = DECAY_EMA
        self.LOG = LOG
        self.SEED = SEED
        self.N_EPOCH = N_EPOCH
        self.N_THREADS = N_THREADS
        self.DROPOUT = DROPOUT
        if MEAN_FILE is not None:
            MEAN_ARRAY = tf.constant(np.load(MEAN_FILE), dtype=tf.float32) # (3)
            self.MEAN_ARRAY = tf.reshape(MEAN_ARRAY, [1, 1, 3])
            self.SUB_MEAN = True
        else:
            self.SUB_MEAN = False

        self.sess = tf.InteractiveSession()

        self.sess.as_default()
        
        self.var_to_reg = []
        self.var_to_sum = []
        self.TF_RECORDS = TF_RECORDS
        self.init_queue(TF_RECORDS)

        self.init_vars()
        self.init_model_architecture()
        self.init_training_graph()
        self.Saver()
        self.DEBUG = DEBUG
        self.loss_func = LOSS_FUNC
        self.weight_decay = WEIGHT_DECAY
    def init_queue(self, tfrecords_filename):
        self.filename_queue = tf.train.string_input_producer(
                              [tfrecords_filename], num_epochs=10)
        with tf.device('/cpu:0'):
            self.image, self.annotation = read_and_decode(self.filename_queue, 
                                                          self.IMAGE_SIZE[0], 
                                                          self.IMAGE_SIZE[1],
                                                          self.BATCH_SIZE,
                                                          self.N_THREADS,
                                                          True,
                                                          self.NUM_CHANNELS)
            #self.annotation = tf.divide(self.annotation, 255.)
        print("Queue initialized")
    def input_node_f(self):
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
        self.labels_queue = self.annotation
        self.labels_PH = tf.placeholder_with_default(self.labels_queue, shape=[None,
                                                                          None, 
                                                                          None,
                                                                          1])

        return self.labels_PH
    def init_training_graph(self):

        with tf.name_scope('Evaluation'):
            # self.logits = self.conv_layer_f(self.last, self.logits_weight, strides=[1,1,1,1], scope_name="logits/")
            with tf.name_scope("logits/"):
                self.logits2 = tf.nn.conv2d(self.last, self.logits_weight, strides=[1,1,1,1], padding="VALID")
                self.logits = tf.nn.bias_add(self.logits2, self.logits_biases)
            self.predictions = self.logits
            #self.predictions = tf.squeeze(self.logits, [3])
            #softmax = tf.nn.softmax(self.logits)
            #print softmax.get_shape()
            #self.predictions = tf.slice(softmax, [0, 0, 0, 0], [-1, -1, -1, 1])
            with tf.name_scope('Loss'):

                self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.logits, self.train_labels_node))
                #self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.predictions, self.train_labels_node))
                tf.summary.scalar("mean_squared_error", self.loss)
            self.predictions = tf.squeeze(self.predictions, [3])
            self.train_prediction = self.predictions

            self.test_prediction = self.predictions

        tf.global_variables_initializer().run()

        print('Computational graph initialised')

    def init_vars(self):

        #### had to add is_training, self.reuse

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

        error = mean_squared_error(labels.flatten(), predictions.flatten())

        return error 

    def Validation(self, DG_TEST, step):
        if DG_TEST is None:
            print "no validation"
        else:
            n_test = DG_TEST.length
            n_batch = int(np.ceil(float(n_test) / self.BATCH_SIZE)) 

            l = 0.
            for i in range(n_batch):
                Xval, Yval = DG_TEST.Batch(0, self.BATCH_SIZE)
                #Yval = Yval / 255.
                feed_dict = {self.input_node: Xval,
                             self.train_labels_node: Yval,
                             self.is_training: False}
                l_tmp, pred, s = self.sess.run([self.loss, 
                                                self.predictions,
                                                self.merged_summary],
                                                feed_dict=feed_dict)
                l += l_tmp
                for j in range(self.BATCH_SIZE):
                    CheckOrCreate('step_{}'.format(step))
                    xval_name = join('step_{}'.format(step), "X_val_{}_{}.png".format(i, j))
                    yval_name = join('step_{}'.format(step), "Y_val_{}_{}.png".format(i, j))
                    pred_name = join('step_{}'.format(step), "pred_{}_{}.png".format(i, j))
                    bin_name = join('step_{}'.format(step), "Y_bin_{}_{}.png".format(i, j))
                    imsave(xval_name, (Xval[j, 92:-92, 92:-92]).astype(np.uint8))
                    imsave(yval_name, Yval[j, :, :, 0].astype(np.uint8))
                    imsave(pred_name, pred[j].astype(np.uint8))
                    GT = Yval[j, :, :, 0].copy()
                    GT[GT > 0] = 255
                    imsave(bin_name, GT.astype(np.uint8))

            l = l / n_batch

            summary = tf.Summary()
            summary.value.add(tag="TestMan/Loss", simple_value=l)
            self.summary_test_writer.add_summary(summary, step) 
            self.summary_test_writer.add_summary(s, step) 
            print('  Validation loss: %.1f' % l)
            self.saver.save(self.sess, self.LOG + '/' + "model.ckpt", step)

    def train(self, DGTest):
        epoch = self.STEPS * self.BATCH_SIZE // self.N_EPOCH
        self.Saver()
        trainable_var = tf.trainable_variables()
        self.LearningRateSchedule(self.LEARNING_RATE, self.K, epoch)
        self.optimization(trainable_var)
        self.ExponentialMovingAverage(trainable_var, self.DECAY_EMA)
        init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
        self.sess.run(init_op)
        self.regularize_model()

        self.Saver()
        
        

        self.summary_test_writer = tf.summary.FileWriter(self.LOG + '/test',
                                            graph=self.sess.graph)

        self.summary_writer = tf.summary.FileWriter(self.LOG + '/train', graph=self.sess.graph)
        self.merged_summary = tf.summary.merge_all()
        steps = self.STEPS

        print "self.global step", int(self.global_step.eval())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
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
                self.Validation(DGTest, step)
        coord.request_stop()
        coord.join(threads)
    def predict(self, tensor):
        feed_dict = {self.input_node: tensor,
                     self.is_training: False}
        pred = self.sess.run(self.predictions,
                            feed_dict=feed_dict)
        return pred

if __name__== "__main__":

    parser = OptionParser()

#    parser.add_option("--gpu", dest="gpu", default="0", type="string",
#                      help="Input file (raw data)")
    parser.add_option("--tf_record", dest="TFRecord", type="string",
                      help="Where to find the TFrecord file")

    parser.add_option("--path", dest="path", type="string",
                      help="Where to collect the patches")

    parser.add_option("--log", dest="log",
                      help="log dir")

    parser.add_option("--learning_rate", dest="lr", type="float",
                      help="learning_rate")

    parser.add_option("--batch_size", dest="bs", type="int",
                      help="batch size")

    parser.add_option("--epoch", dest="epoch", type="int",
                      help="number of epochs")

    parser.add_option("--n_features", dest="n_features", type="int",
                      help="number of channels on first layers")

    parser.add_option("--weight_decay", dest="weight_decay", type="float",
                      help="weight decay value")

    parser.add_option("--dropout", dest="dropout", type="float",
                      default=0.5, help="dropout value to apply to the FC layers.")

    parser.add_option("--mean_file", dest="mean_file", type="str",
                      help="where to find the mean file to substract to the original image.")

    parser.add_option('--n_threads', dest="THREADS", type=int, default=100,
                      help="number of threads to use for the preprocessing.")

    (options, args) = parser.parse_args()


    TFRecord = options.TFRecord
    N_FEATURES = options.n_features
    WEIGHT_DECAY = options.weight_decay
    DROPOUT = options.dropout
    MEAN_FILE = options.mean_file 
    N_THREADS = options.THREADS



    LEARNING_RATE = options.lr
    if int(str(LEARNING_RATE)[-1]) > 7:
        lr_str = "1E-{}".format(str(LEARNING_RATE)[-1])
    else:
        lr_str = "{0:.8f}".format(LEARNING_RATE).rstrip("0")
    SAVE_DIR = options.log + "/" + "{}".format(N_FEATURES) + "_" +"{0:.8f}".format(WEIGHT_DECAY).rstrip("0") + "_" + lr_str

    
    
    HEIGHT = 224 
    WIDTH = 224
    
    
    BATCH_SIZE = options.bs
    LRSTEP = "4epoch"
    SUMMARY = True
    S = SUMMARY
    N_EPOCH = options.epoch

    PATH = options.path


    HEIGHT = 212
    WIDTH = 212
    SIZE = (HEIGHT, WIDTH)

    N_TRAIN_SAVE = 10
 
    CROP = 4


    transform_list, transform_list_test = ListTransform(n_elastic=0)

    DG_TRAIN = DataGenMulti(PATH, split='train', crop = CROP, size=(HEIGHT, WIDTH),
                       transforms=transform_list, num="test", UNet=True, mean_file=None)

    test_patient = ["test"]
    DG_TRAIN.SetPatient(test_patient)
    N_ITER_MAX = N_EPOCH * DG_TRAIN.length // BATCH_SIZE

    DG_TEST  = DataGenMulti(PATH, split="test", crop = 1, size=(500, 500), num="test", 
                       transforms=transform_list_test, UNet=True, mean_file=MEAN_FILE)
    DG_TEST.SetPatient(test_patient)

    model = UNetDistance(TFRecord,    LEARNING_RATE=LEARNING_RATE,
                                       BATCH_SIZE=BATCH_SIZE,
                                       IMAGE_SIZE=SIZE,
                                       NUM_CHANNELS=3,
                                       STEPS=N_ITER_MAX,
                                       LRSTEP=LRSTEP,
                                       N_PRINT=N_TRAIN_SAVE,
                                       LOG=SAVE_DIR,
                                       SEED=42,
                                       WEIGHT_DECAY=WEIGHT_DECAY,
                                       N_FEATURES=N_FEATURES,
                                       N_EPOCH=N_EPOCH,
                                       N_THREADS=N_THREADS,
                                       MEAN_FILE=MEAN_FILE,
                                       DROPOUT=DROPOUT)

    model.train(DG_TEST)