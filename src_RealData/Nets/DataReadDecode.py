#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


def read_and_decode(filename_queue, IMAGE_HEIGHT, IMAGE_WIDTH,
                    BATCH_SIZE, N_THREADS, UNET, CHANNELS=3):
    """
    Read and decode images from TFRecord file.
    """
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    if not UNET:
        features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string)
            })

        height_img = tf.cast(features['height'], tf.int32)
        width_img = tf.cast(features['width'], tf.int32)

        height_mask = height_img
        width_mask = width_img

        const_IMG_HEIGHT = IMAGE_HEIGHT
        const_IMG_WIDTH = IMAGE_WIDTH

        const_MASK_HEIGHT = IMAGE_HEIGHT
        const_MASK_WIDTH = IMAGE_WIDTH


    else:
        features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
            'height_img': tf.FixedLenFeature([], tf.int64),
            'width_img': tf.FixedLenFeature([], tf.int64),
            'height_mask': tf.FixedLenFeature([], tf.int64),
            'width_mask': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string)
            })

        height_img = tf.cast(features['height_img'], tf.int32)
        width_img = tf.cast(features['width_img'], tf.int32)

        height_mask = tf.cast(features['height_mask'], tf.int32)
        width_mask = tf.cast(features['width_mask'], tf.int32)

        const_IMG_HEIGHT = IMAGE_HEIGHT + 184
        const_IMG_WIDTH = IMAGE_WIDTH + 184

        const_MASK_HEIGHT = IMAGE_HEIGHT
        const_MASK_WIDTH = IMAGE_WIDTH


    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    annotation = tf.decode_raw(features['mask_raw'], tf.uint8)
    
    
    image_shape = tf.stack([height_img, width_img, CHANNELS])
    annotation_shape = tf.stack([height_mask, width_mask, 1])
    
    image = tf.reshape(image, image_shape)
    annotation = tf.reshape(annotation, annotation_shape)
    
    image_size_const = tf.constant((const_IMG_HEIGHT, const_IMG_WIDTH, CHANNELS), dtype=tf.int32)
    annotation_size_const = tf.constant((const_MASK_HEIGHT, const_MASK_WIDTH, 1), dtype=tf.int32)
    
    # Random transformations can be put here: right before you crop images
    # to predefined size. To get more information look at the stackoverflow
    # question linked above.
    image_f = tf.cast(image, tf.float32)
    annotation_f = tf.cast(annotation, tf.float32)
    
    resized_image = tf.image.resize_image_with_crop_or_pad(image=image_f,
                                           target_height=const_IMG_HEIGHT,
                                           target_width=const_IMG_WIDTH)
    
    resized_annotation = tf.image.resize_image_with_crop_or_pad(image=annotation_f,
                                           target_height=const_MASK_HEIGHT,
                                           target_width=const_MASK_WIDTH)

    images, annotations = tf.train.shuffle_batch( [resized_image, resized_annotation],
                                                 batch_size=BATCH_SIZE,
                                                 capacity=10 + 3 * BATCH_SIZE,
                                                 num_threads=N_THREADS,
                                                 min_after_dequeue=10)
    
    return images, annotations
