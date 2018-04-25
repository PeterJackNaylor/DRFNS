#!/usr/bin/env python
# -*- coding: utf-8 -*-

from skimage.morphology import watershed
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import reconstruction, dilation, erosion, disk, diamond, square
from skimage import img_as_ubyte


def PrepareProb(img, uint8=True):
    """
    Prepares the prob image for post-processing, it can convert from
    float -> to uint8 and it can inverse it if needed.
    """
    if uint8:
        img = img_as_ubyte(img)
        img = 255 - img
    else:
        img = img.max() - img
    return img


def HreconstructionErosion(prob_img, h, uint8=True):
    """
    Performs a H minimma reconstruction via an erosion method.
    """
    maxi = 255 if uint8 else prob_img.max()
    def making_top_mask(x, lamb=h):
        return min(maxi, x + lamb)
    f = np.vectorize(making_top_mask)
    shift_prob_img = f(prob_img)

    seed = shift_prob_img
    mask = prob_img
    recons = reconstruction(
        seed, mask, method='erosion').astype(prob_img.dtype)
    return recons


def find_maxima(img, convertuint8=False, inverse=False, mask=None):
    """
    Finds all local maxima from 2D image.
    """
    recons = HreconstructionErosion(img, 1)
    if mask is None:
        return recons - img
    else:
        res = recons - img
        res[mask==0] = 0
        return res
def GetContours(img):
    """
    Returns only the contours of the image.
    The image has to be a binary image 
    """
    img[img > 0] = 1
    return dilation(img, disk(2)) - erosion(img, disk(2))


def generate_wsl(ws):
    """
    Generates watershed line that correspond to areas of
    touching objects.
    """
    se = square(3)
    ero = ws.copy()
    ero[ero == 0] = ero.max() + 1
    ero = erosion(ero, se)
    ero[ws == 0] = 0

    grad = dilation(ws, se) - ero
    grad[ws == 0] = 0
    grad[grad > 0] = 255
    grad = grad.astype(np.uint8)
    return grad

def assign_wsl(label_res, wsl):
    """
    Assigns wsl to biggest connect component in the bbox around the wsl.
    """
    wsl_lb = label(wsl)
    objects = regionprops(wsl_lb)
    for obj in objects:
        x_b, y_b, x_t, y_t = obj.bbox
        import pdb; pdb.set_trace()
        val, counts = np.unique(label_res[x_b:x_t, y_b:y_t], return_counts=True)
        if 0 in val:
	    coord = np.where(val == 0)[0]
            counts[coord] = 0
        if obj.label in val:
            coord = np.where(val == obj.label)[0]
            counts[coord] = 0
        best_lab = val[counts.argmax()]
        label_res[wsl_lb == obj.label] = best_lab
    return label_res
         



def DynamicWatershedAlias(p_img, lamb, p_thresh = 0.5, uint8=True):
    """
    Applies our dynamic watershed to 2D prob/dist image.
    """
    b_img = (p_img > p_thresh) + 0
    Probs_inv = PrepareProb(p_img, uint8)


    Hrecons = HreconstructionErosion(Probs_inv, lamb)
    markers_Probs_inv = find_maxima(Hrecons, mask = b_img)
    markers_Probs_inv = label(markers_Probs_inv)
    ws_labels = watershed(Hrecons, markers_Probs_inv, mask=b_img)
    result = ArrangeLabel(ws_labels)
    #wsl = generate_wsl(arrange_label)
    #result = assign_wsl(arrange_label, wsl)

    return result

def ArrangeLabel(mat):
    """
    Arrange label image as to effectively put background to 0.
    """
    val, counts = np.unique(mat, return_counts=True)
    background_val = val[np.argmax(counts)]
    mat = label(mat, background = background_val)
    if np.min(mat) < 0:
        mat += np.min(mat)
        mat = ArrangeLabel(mat)
    return mat


def PostProcess(prob_image, param=7, thresh = 0.5):
    """
    Perform DynamicWatershedAlias with some default parameters.
    """
    segmentation_mask = DynamicWatershedAlias(prob_image, param, thresh)
    return segmentation_mask

def PostProcessFloat(prob_image, param=0, thresh = 0.):
    """
    Perform DynamicWatershedAlias with some default parameters.
    """
    segmentation_mask = DynamicWatershedAlias(prob_image, param, thresh, uint8=False)
    return segmentation_mask
