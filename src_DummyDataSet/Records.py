#!/usr/bin/env python
# -*- coding: utf-8 -*-

from optparse import OptionParser
from Data.ImageTransform import ListTransform
from Data.CreateTFRecords import CreateTFRecord

def options_parser():

    parser = OptionParser()

    parser.add_option('--output', dest="TFRecords", type="string",
                      help="name for the output .tfrecords")
    parser.add_option('--path', dest="path", type="str",
                      help="Where to find the annotations")
    parser.add_option('--crop', dest="crop", type="int",
                      help="Number of crops to divide one image in")
    # parser.add_option('--UNet', dest="UNet", type="bool",
    #                   help="If image and annotations will have different shapes")
    parser.add_option('--size', dest="size", type="int",
                      help='first dimension for size')
    parser.add_option('--seed', dest="seed", type="int", default=42,
                      help='Seed to use, still not really implemented')  
    parser.add_option('--epoch', dest="epoch", type ="int",
                       help="Number of epochs to perform")  
    parser.add_option('--type', dest="type", type ="str",
                       help="Type for the datagen")  
    parser.add_option('--UNet', dest='UNet', action='store_true')
    parser.add_option('--no-UNet', dest='UNet', action='store_false')

    parser.add_option('--train', dest='split', action='store_true')
    parser.add_option('--test', dest='split', action='store_false')
    parser.set_defaults(feature=True)

    (options, args) = parser.parse_args()
    options.SIZE = (options.size, options.size)
    return options

if __name__ == '__main__':

    options = options_parser()

    OUTNAME = options.TFRecords
    PATH = options.path
    CROP = options.crop
    SIZE = options.SIZE
    SPLIT = "train" if options.split else "test"
    transform_list, transform_list_test = ListTransform() 
    TRANSFORM_LIST = transform_list
    UNET = options.UNet
    SEED = options.seed
    TEST_PATIENT = ["test"]
    N_EPOCH = options.epoch
    TYPE = options.type
    

    CreateTFRecord(OUTNAME, PATH, CROP, SIZE,
                   TRANSFORM_LIST, UNET, None, 
                   SEED, TEST_PATIENT, N_EPOCH,
                   TYPE=TYPE, SPLIT=SPLIT)