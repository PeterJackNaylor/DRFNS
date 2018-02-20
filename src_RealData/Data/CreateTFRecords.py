#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from DataGenRandomT import DataGenRandomT
from DataGenClass import DataGen3, DataGenMulti, DataGen3reduce
import numpy as np

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def CreateTFRecord(OUTNAME, PATH, CROP, SIZE,
                   TRANSFORM_LIST, UNET, MEAN_FILE, 
                   SEED, TEST_PATIENT, N_EPOCH, TYPE = "Normal",
                   SPLIT="train"):
    """
    Takes a DataGen object and creates an associated TFRecord file. 
    We do not perform data augmentation on the fly but save the 
    augmented images in the record. Most of the parameters here 
    reference paramaters of the DataGen object. In particular, PATH,
    CROP, SIZE, TRANSFORM_LIST, UNET, SEED and TEST_PATIENT. 
    OUTNAME is the name of the record.
    """

    tfrecords_filename = OUTNAME
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    
    if TYPE == "Normal":
        DG = DataGenRandomT(PATH, split=SPLIT, crop=CROP, size=SIZE,
                        transforms=TRANSFORM_LIST, UNet=UNET, num=TEST_PATIENT,
                        mean_file=MEAN_FILE, seed_=SEED)

    elif TYPE == "3class":
        DG = DataGen3(PATH, split=SPLIT, crop = CROP, size=SIZE, 
                       transforms=TRANSFORM_LIST, UNet=UNET, num=TEST_PATIENT,
                       mean_file=MEAN_FILE, seed_=SEED)
    elif TYPE == "ReducedClass":
        DG = DataGen3reduce(PATH, split=SPLIT, crop = CROP, size=SIZE, 
                       transforms=TRANSFORM_LIST, UNet=UNET, num=TEST_PATIENT,
                       mean_file=MEAN_FILE, seed_=SEED)
    elif TYPE == "JUST_READ":
        DG = DataGenMulti(PATH, split=SPLIT, crop = CROP, size=SIZE, 
                       transforms=TRANSFORM_LIST, UNet=UNET, num=TEST_PATIENT,
                       mean_file=MEAN_FILE, seed_=SEED)

    DG.SetPatient(TEST_PATIENT)
    N_ITER_MAX = N_EPOCH * DG.length

    original_images = []
    key = DG.RandomKey(False)
    if not UNET:
      for _ in range(N_ITER_MAX):
          key = DG.NextKeyRandList(0)
          img, annotation = DG[key]
  #        img = img.astype(np.uint8)
          annotation = annotation.astype(np.uint8)
          height = img.shape[0]
          width = img.shape[1]
      
          original_images.append((img, annotation))
          
          img_raw = img.tostring()
          annotation_raw = annotation.tostring()
          
          example = tf.train.Example(features=tf.train.Features(feature={
              'height': _int64_feature(height),
              'width': _int64_feature(width),
              'image_raw': _bytes_feature(img_raw),
              'mask_raw': _bytes_feature(annotation_raw)}))
          
          writer.write(example.SerializeToString())
    else:
      for _ in range(N_ITER_MAX):
          key = DG.NextKeyRandList(0)
          img, annotation = DG[key]
  #        img = img.astype(np.uint8)
          annotation = annotation.astype(np.uint8)
          height_img = img.shape[0]
          width_img = img.shape[1]

          height_mask = annotation.shape[0]
          width_mask = annotation.shape[1]
      
          original_images.append((img, annotation))
          
          img_raw = img.tostring()
          annotation_raw = annotation.tostring()
          
          example = tf.train.Example(features=tf.train.Features(feature={
              'height_img': _int64_feature(height_img),
              'width_img': _int64_feature(width_img),
              'height_mask': _int64_feature(height_mask),
              'width_mask': _int64_feature(width_mask),
              'image_raw': _bytes_feature(img_raw),
              'mask_raw': _bytes_feature(annotation_raw)}))
          
          writer.write(example.SerializeToString())


    writer.close()
