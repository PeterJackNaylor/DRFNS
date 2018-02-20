#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import copy
from tf_image_segmentation.models.fcn_8s import FCN_8s
from tf_image_segmentation.utils.tf_records import read_tfrecord_and_decode_into_image_annotation_pair_tensors
from tf_image_segmentation.utils.training import get_valid_logits_and_labels
from tf_image_segmentation.utils.inference import adapt_network_for_any_size_input
from tf_image_segmentation.utils.visualization import visualize_segmentation_adaptive
from tf_image_segmentation.utils.augmentation import (distort_randomly_image_color,
                                                      flip_randomly_left_right_image_with_annotation,
                                                      scale_randomly_image_with_annotation_with_fixed_size_output)
import os
from glob import glob
import numpy as np
from scipy import misc
slim = tf.contrib.slim
from utils import ComputeMetrics

class FCN8():
    """
    FCN8 object for performing training, testing and validating.
    """
    def __init__(self, checkpoint, save_dir, record, 
                       size, num_labels, n_print, split='train'):

        self.checkpoint8 = checkpoint
        self.Setuped = False
        self.checkpointnew = save_dir
        self.record = record
        self.size = size
        self.num_labels = num_labels
        self.n_print = n_print
        self.setup_record()
        self.class_labels = range(num_labels)
        self.class_labels.append(255)
        if split != "validation":
            self.fcn_8s_checkpoint_path = glob(self.checkpoint8 + "/*.data*")[0].split(".data")[0]
        else:
            self.setup_val(record)
    def setup_record(self):
        """
        Setup record reading.
        """
        filename_queue = tf.train.string_input_producer(
                                    [self.record], num_epochs=10)

        self.image, self.annotation = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)
        self.resized_image, resized_annotation = scale_randomly_image_with_annotation_with_fixed_size_output(self.image, self.annotation, (self.size, self.size))
        self.resized_annotation = tf.squeeze(resized_annotation)
    def setup_train8(self, lr):
        """
        Setups queues and model evaluation.
        """


        image_batch, annotation_batch = tf.train.shuffle_batch( [self.resized_image, self.resized_annotation],
                                                     batch_size=1,
                                                     capacity=3000,
                                                     num_threads=2,
                                                     min_after_dequeue=1000)


        upsampled_logits_batch, fcn_8s_variables_mapping = FCN_8s(image_batch_tensor=image_batch,
                                                                   number_of_classes=self.num_labels,
                                                                   is_training=True)

        valid_labels_batch_tensor, valid_logits_batch_tensor = get_valid_logits_and_labels(annotation_batch_tensor=annotation_batch,
                                                                                           logits_batch_tensor=upsampled_logits_batch,
                                                                                           class_labels=self.class_labels)


        # Count true positives, true negatives, false positives and false negatives.
        actual = tf.contrib.layers.flatten(tf.cast(annotation_batch, tf.int64))

        self.predicted_img = tf.argmax(upsampled_logits_batch, axis=3)
        cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits_batch_tensor,
                                                                  labels=valid_labels_batch_tensor)
        self.cross_entropy_sum = tf.reduce_mean(cross_entropies)

        pred = tf.argmax(upsampled_logits_batch, dimension=3)

        probabilities = tf.nn.softmax(upsampled_logits_batch)


        with tf.variable_scope("adam_vars"):
            self.train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.cross_entropy_sum)


        # Variable's initialization functions

        self.init_fn = slim.assign_from_checkpoint_fn(model_path=self.fcn_8s_checkpoint_path,
                                                 var_list=fcn_8s_variables_mapping)

        global_vars_init_op = tf.global_variables_initializer()

        self.merged_summary_op = tf.summary.merge_all()

        self.summary_string_writer = tf.summary.FileWriter("./log_8_{}".format(lr))

        if not os.path.exists(self.checkpointnew):
             os.makedirs(self.checkpointnew)
            
        #The op for initializing the variables.
        local_vars_init_op = tf.local_variables_initializer()

        self.combined_op = tf.group(local_vars_init_op, global_vars_init_op)

        # We need this to save only model variables and omit
        # optimization-related and other variables.
        model_variables = slim.get_model_variables()
        self.saver = tf.train.Saver(model_variables)

    def train8(self, iters, lr):
        """
        Trains the model.
        """


        self.setup_train8(lr)
        model_save = os.path.join(self.checkpointnew, self.checkpointnew)
        with tf.Session()  as sess:
            
            sess.run(self.combined_op)
            self.init_fn(sess)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            
            # 10 epochs
            for i in xrange(0, iters):
                cross_entropy, summary_string, _ = sess.run([self.cross_entropy_sum,
                                              self.merged_summary_op,
                                              self.train_step ])
                
                if i % self.n_print == 0:
                    self.summary_string_writer.add_summary(summary_string, i)
                    save_path = self.saver.save(sess, model_save)
                    print("Model saved in file: %s" % save_path)
                    
                
            coord.request_stop()
            coord.join(threads)
            
            save_path = self.saver.save(sess, model_save)
            print("Model saved in file: %s" % save_path)


    def test8(self, steps, restore, p1):
        """
        Tests the model.
        """
        # Fake batch for image and annotation by adding
        # leading empty axis.
        image_batch_tensor = tf.expand_dims(self.image, axis=0)
        annotation_batch_tensor = tf.expand_dims(self.annotation, axis=0)
        # Be careful: after adaptation, network returns final labels
        # and not logits
        FCN_8s_bis = adapt_network_for_any_size_input(FCN_8s, 32)


        pred, fcn_16s_variables_mapping = FCN_8s_bis(image_batch_tensor=image_batch_tensor,
                                                  number_of_classes=self.num_labels,
                                                  is_training=False)
        prob = [h for h in [s for s in [t for t in pred.op.inputs][0].op.inputs][0].op.inputs][0]

        initializer = tf.local_variables_initializer()
        saver = tf.train.Saver()
        loss, roc = 0., 0.
        acc, F1, recall = 0., 0., 0.
        precision, jac, AJI = 0., 0., 0.
        with tf.Session() as sess:
            
            sess.run(initializer)
            saver.restore(sess, restore)
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            
            
            for i in xrange(steps):
                    
                image_np, annotation_np, pred_np, prob_np = sess.run([self.image, self.annotation, pred, prob])
                prob_float = np.exp(-prob_np[0,:,:,0]) / (np.exp(-prob_np[0,:,:,0]) + np.exp(-prob_np[0,:,:,1]))
                prob_int8 = misc.imresize(prob_float, size=image_np[:,:,0].shape)

                prob_float = (prob_int8.copy().astype(float) / 255)
                out = ComputeMetrics(prob_float, annotation_np[:,:,0], p1, 0.5)
                acc += out[0]
                roc += out[1]
                jac += out[2]
                recall += out[3]
                precision += out[4]
                F1 += out[5]
                AJI += out[6]
            coord.request_stop()
            coord.join(threads)
            loss, acc, F1 = np.array([loss, acc, F1]) / steps
            recall, precision, roc = np.array([recall, precision, roc]) / steps
            jac, AJI = np.array([jac, AJI]) / steps
            return loss, acc, F1, recall, precision, roc, jac, AJI 
    def setup_val(self, tfname):
        """
        Setups the model in case we need to validate.
        """
        self.restore = glob(os.path.join(self.checkpoint8, "FCN__*", "*.data*" ))[0].split(".data")[0]  
        
        filename_queue = tf.train.string_input_producer(
                                    [tfname], num_epochs=10)
        self.image_queue, self.annotation_queue = read_tfrecord_and_decode_into_image_annotation_pair_tensors(filename_queue)
        self.image = tf.placeholder_with_default(self.image, shape=[None, 
                                                                    None,
                                                                       3])
        self.annotation = tf.placeholder_with_default(self.annotation_queue, shape=[None,
                                                                    None,
                                                                       1])
        self.resized_image, resized_annotation = scale_randomly_image_with_annotation_with_fixed_size_output(self.image, self.annotation, (self.size, self.size))
        self.resized_annotation = tf.squeeze(resized_annotation)
        image_batch_tensor = tf.expand_dims(self.image, axis=0)
        annotation_batch_tensor = tf.expand_dims(self.annotation, axis=0)
        # Be careful: after adaptation, network returns final labels
        # and not logits
        FCN_8s_bis = adapt_network_for_any_size_input(FCN_8s, 32)
        self.pred, fcn_16s_variables_mapping = FCN_8s_bis(image_batch_tensor=image_batch_tensor,
                                                      number_of_classes=self.num_labels,
                                                      is_training=False)
        self.prob = [h for h in [s for s in [t for t in self.pred.op.inputs][0].op.inputs][0].op.inputs][0]
        initializer = tf.local_variables_initializer()
        self.saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(initializer)
            self.saver.restore(sess, self.restore)
    def validation(self, DG_TEST, steps, p1, p2, save_organ):
        """
        Validates the model.
        """
        tmp_name = os.path.basename(save_organ) + ".tfRecord" 
        res = []
        print "DOing this {}".format(save_organ)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            self.saver.restore(sess, self.restore)
            #coord = tf.train.Coordinator()
            #threads = tf.train.start_queue_runners(coord=coord)
            for i in xrange(steps):
                Xval, Yval = DG_TEST.Batch(0, 1)

                feed_dict = {self.image: Xval[0], self.annotation: Yval[0]}
                image_np, annotation_np, pred_np, prob_np = sess.run([self.image, self.annotation, self.pred, self.prob], feed_dict=feed_dict)
                prob_float = np.exp(-prob_np[0,:,:,0]) / (np.exp(-prob_np[0,:,:,0]) + np.exp(-prob_np[0,:,:,1]))
                prob_int8 = misc.imresize(prob_float, size=image_np[:,:,0].shape)

                prob_float = (prob_int8.copy().astype(float) / 255)
                out = ComputeMetrics(prob_float, annotation_np[:,:,0], p1, 0.5, rgb=image_np, save_path=save_organ, ind=i)
                out = [0.] + list(out)
                res.append(out)
            return res
    def copy(self):
        """
        Copies itself and returns a copied version of himself.
        """
        return copy.deepcopy(self) 
