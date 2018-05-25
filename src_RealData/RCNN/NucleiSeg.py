"""
This code was revised from this report:


Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018/
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a new model starting from ImageNet weights
    python3 NucleiSeg.py train --dataset=/path/to/dataset --subset=train --weights=imagenet
    # Train a new model starting from specific weights file
    python3 NucleiSeg.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5
    # Resume training a model that you had trained earlier
    python3 NucleiSeg.py train --dataset=/path/to/dataset --subset=train --weights=last
    # Generate submission file
    python3 NucleiSeg.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa
from glob import glob
from skimage.io import imsave
# Root directory of the project
ROOT_DIR = os.path.abspath("Mask_RCNN/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/nucleus/")

# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to surve as a validation set.
N_VAL_IMAGE_IDS = 4 * 4
N_TRAIN_ = 12 * 4 
MEAN_FILE = "mean_file.npy"
############################################################
#  Configurations
############################################################

class NucleiSegConfig(Config):   
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    LEARNING_RATE = 0.001
    NAME = "NucleiSeg"
    GPU_COUNT = 4
    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + nucleus

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = N_TRAIN_ // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, N_VAL_IMAGE_IDS // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0.9

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet101"
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0
    RPN_ANCHOR_RATIOS = [0.5, 1., 2.]
    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.5

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64 * 4
    
    # Image mean (RGB)
#    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])
#    MEAN_PIXEL = np.load(MEAN_FILE).astype(float)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    IMAGE_MIN_SCALE = 0
    
    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (28, 28)  # (height, width) of the mini-mask
   
    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 600
    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 256
    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 512
    DETECTION_NMS_THRESHOLD = 0.2

class NucleiSegInferenceConfig(NucleiSegConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_MAX_INSTANCES = 5000


############################################################
#  Dataset
############################################################

class NucleiSegDataset(utils.Dataset):

    def load_nucleus(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.
        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("NucleiSeg", 1, "NucleiSeg")

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val", "test"]
        subset_dir = "stage1_train" if subset in ["train", "val"] else subset
        if subset == "train":
            organ = ["trainbreast", "trainkidney", "trainliver", "trainprostate"]
        elif subset == "val":
            organ = ["validation"]
        else:
            organ = ["testbreast", "testkidney", "testliver", "testprostate", \
                     "bladder", "colorectal", "stomach"]
        image_ids = []
        for o in organ:
            image_ids += glob(os.path.join(dataset_dir, "Slide_" + o, "*.png"))
        # Add images
        for image_id in image_ids:
            self.add_image(
                "NucleiSeg",
                image_id=image_id,
                path=image_id)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_file = info['path'].replace('Slide', 'GT')
        mask = skimage.measure.label(skimage.io.imread(mask_file))
        res = np.stack([mask == i for i in range(1, mask.max() + 1)], axis=-1)
        return res, np.ones([res.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "NucleiSeg":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = NucleiSegDataset()
    dataset_train.load_nucleus(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = NucleiSegDataset()
    dataset_val.load_nucleus(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=20,
                augmentation=augmentation,
                layers='all')

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 100,
                epochs=20,
                augmentation=augmentation,
                layers='all')


############################################################
#  Detection
############################################################
def FromMaskToPred(mask):
    x, y, z = mask.shape
    res = np.zeros(shape=(x, y), dtype='int64')
    for val, k in enumerate(range(z)):
        res[mask[:,:,k]] = val + 1
    return res

import os

def CheckOrCreate(path):
    """
    If path exists, does nothing otherwise it creates it.
    """
    if not os.path.isdir(path):
        os.makedirs(path)
from skimage.morphology import watershed, dilation, disk, reconstruction

def my_clear_border(pred, buffer_size, x, y, xw, yh, image):
    sx, sy = image.shape[0:2]
    pred_cpy = pred.copy()
    pred_cpy[pred_cpy > 0] = 1
    mask = pred_cpy
    border_fill = np.zeros_like(pred_cpy)
    # import pdb; pdb.set_trace()
    if x != 0:
        border_fill[0:buffer_size, :] = 1
        mask[0:buffer_size, :] = 1
    if y != 0:
        border_fill[:, 0:buffer_size] = 1
        mask[:, 0:buffer_size] = 1
    if xw != sx:
        border_fill[-buffer_size::, :] = 1
        mask[-buffer_size::, :] = 1
    if yh != sy:
        border_fill[:,-buffer_size::] = 1
        mask[:,-buffer_size::] = 1
    
    reconstructed = reconstruction(border_fill, mask, 'dilation')
    pred[reconstructed > 0] = 0
    return skimage.measure.label(pred)

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "test_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = NucleiSegDataset()
    dataset.load_nucleus(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        # import pdb; pdb.set_trace()
        image = dataset.load_image(image_id)
        windowSize = (1000, 1000)
        stepSize = (250, 250)
        res = np.zeros(shape=(1000, 1000), dtype="int64")
        nmax = 0
        for x ,y, xw, yh, sub_image in sliding_window(image, stepSize, windowSize): 
            # Detect objects
            pred = model.detect([sub_image], verbose=0)[0]
            # Encode image to RLE. Returns a string of multiple lines
            source_id = dataset.image_info[image_id]["id"]
            PRED = FromMaskToPred(pred['masks'])
            nmax = PRED.max()
            #buffer_size = 5
            #PRED = my_clear_border(PRED, buffer_size, x, y, xw, yh, image)
            PRED[PRED > 0] += nmax
            res[x:xw, y:yh] += PRED
            nmax += PRED.max()
            visualize.display_instances(
                sub_image, pred['rois'], pred['masks'], pred['class_ids'],
                dataset.class_names, pred['scores'],
                show_bbox=False, show_mask=False,
                title="Predictions")
            plt.savefig("{}/{}".format(submit_dir, os.path.basename(dataset.image_info[image_id]["id"])).replace('.png', str(x)+str(y)+".png"))
            
        file_name = "{}/{}".format(submit_dir, os.path.basename(dataset.image_info[image_id]["id"])).replace('.png', '_mask.png')
        CheckOrCreate(os.path.dirname(file_name))
        imsave(file_name, res)
        # Save image with masks

    # Save to csv file
    print("Saved to ", submit_dir)


def sliding_window(image, stepSize, windowSize):
    # slide a window across the imag
    for x in range(0, image.shape[0] - windowSize[0] + stepSize[0], stepSize[0]):
        for y in range(0, image.shape[1] - windowSize[1] + stepSize[1], stepSize[1]):
            # yield the current window  
            res_img = image[x:x + windowSize[0], y:y + windowSize[1]]
            change = False
            if res_img.shape[0] != windowSize[0]:
                x = image.shape[0] - windowSize[0]
                change = True
            if res_img.shape[1] != windowSize[1]:
                y = image.shape[1] - windowSize[1]
                change = True
            if change:
                res_img = image[x:x + windowSize[0], y:y + windowSize[1]]
            yield (x, y, x + windowSize[0], y + windowSize[1], res_img)


############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    parser.add_argument('--lr', required=False, default=0.001, type=float,
                        metavar="learning_rate value",
                        help="learning rate")
    parser.add_argument('--DMC', required=False, default=0.9, type=float,
                        metavar="DETECTION_MIN_CONFIDENCE",
                        help="DETECTION_MIN_CONFIDENCE")
    parser.add_argument('--RPN_NMS_THRESHOLD', required=False, default=0.7, type=float,
                        metavar="RPN_NMS_THRESHOLD",
                        help="RPN_NMS_THRESHOLD")
    args = parser.parse_args()
    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = NucleiSegConfig()
        config.LEARNING_RATE = args.lr
    else:
        config = NucleiSegInferenceConfig()
    config.DETECTION_MIN_CONFIDENCE = args.DMC
    config.RPN_NMS_THRESHOLD = args.RPN_NMS_THRESHOLD
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
"Use 'train' or 'detect'".format(args.command))
