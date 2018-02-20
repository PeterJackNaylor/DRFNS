#!/usr/bin/env python
# -*- coding: utf-8 -*-

from DataGenRandomT import DataGenRandomT
import nibabel as ni
from skimage import measure
from scipy.ndimage.morphology import morphological_gradient
from scipy import misc
from utils import generate_wsl

def Contours(bin_image, contour_size=3):
    """
    Finds contours of binary images.
    """
    grad = morphological_gradient(bin_image, size=(contour_size, contour_size))
    return grad

class DataGen3(DataGenRandomT):
    """
    DG object of 3 class object. Background, inner cell and contour of cell.
    """
    def LoadGT(self, path):
        image = ni.load(path)
        img = image.get_data()
        img = measure.label(img)
        wsl = generate_wsl(img[:,:,0])
        img[ img > 0 ] = 1
        wsl[ wsl > 0 ] = 1
        img[:,:,0] = img[:,:,0] - wsl
        if len(img.shape) == 3:
            img = img[:, :, 0].transpose()
        else:
            img = img.transpose()
        cell_border = Contours(img, contour_size=3)
        img[cell_border > 0] = 2
        return img

class DataGenMulti(DataGenRandomT):
    """
    DG object that just reads png from files.
    """
    def LoadGT(self, path):
        image = misc.imread(path.replace(".nii.gz", ".png"))
        return image

class DataGen3reduce(DataGenRandomT):
    """
    DG object that aggregates classes above 4 to be 5.
    """
    def LoadGT(self, path):
        image = misc.imread(path.replace(".nii.gz", ".png"))
        image[image > 4] = 5
        return image