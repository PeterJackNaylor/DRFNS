import os, sys
import pdb

import numpy as np

import vigra
from rdflib.plugins.parsers.pyRdfa.transform.prototype import pref
from Finder.Type_Definitions import preferences

test_in_folder = '/Users/twalter/data/FIMM_histopath/test_in'
test_out_folder = '/Users/twalter/data/FIMM_histopath/test_out'

import skimage
import skimage.io

from skimage.feature import blob_doh
from skimage.morphology import disk
from skimage import morphology
from skimage.filters import rank
from skimage import filters
from skimage import color
from skimage import restoration
from skimage.measure import label
from skimage import measure
from skimage.morphology import watershed
from skimage.feature import peak_local_max

from scipy import ndimage as ndi


from math import sqrt

import matplotlib.pyplot as plt

from cecog import ccore

class Segmentation(object):
    def __init__(self):
        print 'SEGMENTATION'
        if not os.path.exists(test_out_folder):
            os.makedirs(test_out_folder)
            print 'made %s' % test_out_folder
        self.image_name = 'H1'
        return
    
    def read_H_image(self, image_name='H1'):
        filename = os.path.join(test_in_folder, '%s.png' % image_name)
        img = skimage.io.imread(filename)
        self.image_name = image_name
        return img
    
    def overlay(self, img, imbin, contour=False):
        colim = color.gray2rgb(img)
        colorvalue = (0, 100, 200)
        if contour:
            se = morphology.diamond(2)
            ero = morphology.erosion(imbin, se)
            grad = imbin - ero
            colim[grad > 0] = colorvalue
        else:
            colim[imbin>0] = colorvalue
            
        return colim
    
    def output_blob_detection(self, img, blobs):
        colim = color.gray2rgb(img)
        
        for blob in blobs:
            x, y, r = blob
                        
            rr, cc = skimage.draw.circle(x,y,r)
            colorvalue = (255, 0, 0)
            
            if np.min(rr) < 0 or np.min(cc) < 0 or np.max(rr) >= img.shape[0]  or np.max(cc) >= img.shape[1]:
                continue
            
            for i, col in enumerate(colorvalue):
                colim[rr,cc,i] = col
 
        return colim
    
    def blob_detection(self, img):
        blobs = blob_doh(img, max_sigma=80, threshold=.001)
        return blobs
    
    def difference_of_gaussian(self, imin, bigsize=30.0, smallsize=3.0):
        g1 = filters.gaussian_filter(imin, bigsize)
        g2 = filters.gaussian_filter(imin, smallsize)
        diff = 255*(g1 - g2)

        diff[diff < 0] = 0.0
        diff[diff > 255.0] = 255.0
        diff = diff.astype(np.uint8) 
               
        return diff
    
    def test_dog(self, bigsize=30.0, smallsize=3.0, addon='none'):
        img = self.read_H_image()
        diff = self.difference_of_gaussian(img, bigsize, smallsize)
        filename = os.path.join(test_out_folder, 'dog_%s_%s.png' % (self.image_name, addon))
        skimage.io.imsave(filename, diff)

        diff = self.difference_of_gaussian(-img, bigsize, smallsize)
        filename = os.path.join(test_out_folder, 'invdog_%s_%s.png' % (self.image_name, addon))
        skimage.io.imsave(filename, diff)

        return
    
    def test_blob_detection(self):
        img = self.read_H_image()
        blobs = self.blob_detection(img)
        color_out = self.output_blob_detection(img, blobs)
        filename = os.path.join(test_out_folder, 'blobs_%s.png' % self.image_name)
        skimage.io.imsave(filename, color_out)
        
        return
    
    def morpho_rec(self, img, size=10):
        # internal gradient of the cells: 
        se = morphology.diamond(size)
        ero = morphology.erosion(img, se)
        rec = morphology.reconstruction(ero, img, method='dilation').astype(np.dtype('uint8'))
                
        return rec

    def morpho_rec2(self, img, size=10):
        # internal gradient of the cells: 
        se = morphology.diamond(size)
        dil = morphology.dilation(img, se)
        rec = morphology.reconstruction(dil, img, method='erosion').astype(np.dtype('uint8'))
                
        return rec
    
    def test_morpho(self):
    
        img = self.read_H_image()
        
        pref = self.morpho_rec(img, 10)
        filename = os.path.join(test_out_folder, 'rec_%s.png' % self.image_name)
        skimage.io.imsave(filename, pref)

        diff = self.difference_of_gaussian(img, 50.0, 2.0)
        filename = os.path.join(test_out_folder, 'recdog_%s_%s.png' % (self.image_name, '40_1'))
        skimage.io.imsave(filename, diff)

        return 


    def test_morpho2(self, bigsize=20.0, smallsize=3.0, threshold=5.0):
    
        img = self.read_H_image()

        pref = self.morpho_rec(img, 10)
        filename = os.path.join(test_out_folder, 'morpho_00_rec_%s.png' % self.image_name)
        skimage.io.imsave(filename, pref)

        res = self.difference_of_gaussian(pref, bigsize, smallsize)
        filename = os.path.join(test_out_folder, 'morpho_01_diff_%s_%i_%i.png' % (self.image_name, int(bigsize), int(smallsize)))
        skimage.io.imsave(filename, res)
        
        #res = self.morpho_rec2(diff, 15)
        #filename = os.path.join(test_out_folder, 'morpho_02_rec_%s.png' % self.image_name)
        #skimage.io.imsave(filename, res)

        res[res>threshold] = 255
        filename = os.path.join(test_out_folder, 'morpho_03_res_%s_%i.png' % (self.image_name, threshold))
        skimage.io.imsave(filename, res)
        
        se = morphology.diamond(3)
        ero = morphology.erosion(res, se)
        filename = os.path.join(test_out_folder, 'morpho_03_ero_%s_%i.png' % (self.image_name, threshold))
        skimage.io.imsave(filename, ero)
        res[ero>0] = 0
        
        overlay_img = self.overlay(img, res)
        filename = os.path.join(test_out_folder, 'morpho_04_overlay_%s_%i.png' % (self.image_name, int(threshold)))
        skimage.io.imsave(filename, overlay_img)
        
        return 

    def get_rough_detection(self, img, bigsize=40.0, smallsize=4.0, thresh = 0):
        diff = self.difference_of_gaussian(-img, bigsize, smallsize)
        diff[diff>thresh] = 1
        
        se = morphology.square(4)
        ero = morphology.erosion(diff, se)
        
        labimage = label(ero)
        #rec = morphology.reconstruction(ero, img, method='dilation').astype(np.dtype('uint8'))
        
        # connectivity=1 corresponds to 4-connectivity.
        morphology.remove_small_objects(labimage, min_size=600, connectivity=1, in_place=True)
        #res = np.zeros(img.shape)
        ero[labimage==0] = 0
        ero = 1 - ero
        labimage = label(ero)
        morphology.remove_small_objects(labimage, min_size=400, connectivity=1, in_place=True)
        ero[labimage==0] = 0
        res = 1 - ero
        res[res>0] = 255
        
        #temp = 255 - temp
        #temp = morphology.remove_small_objects(temp, min_size=400, connectivity=1, in_place=True)
        #res = 255 - temp
        
        return res
    
    def test_rough_detection(self, bigsize=40.0, smallsize=4.0, thresh = 0):
        
        img = self.read_H_image()
        rough = self.get_rough_detection(img, bigsize, smallsize, thresh)
        
        colorim = self.overlay(img, rough)
        filename = os.path.join(test_out_folder, 'roughdetection_%s.png' % (self.image_name))
        skimage.io.imsave(filename, colorim)
        filename = os.path.join(test_out_folder, 'roughdetection_original_%s.png' % (self.image_name))
        skimage.io.imsave(filename, img)

        return
    
    def prefilter(self, img, rec_size=20, se_size=3):
    
        se = morphology.disk(se_size)
        
        im1 = self.morpho_rec(img, rec_size)
        im2 = self.morpho_rec2(im1, int(rec_size / 2))
        
        im3 = morphology.closing(im2, se)
        
        return im3

    def prefilter_new(self, img, rec_size=20, se_size=3):
    
        img_cc = ccore.numpy_to_image(img, copy=True)        
        im1 = ccore.diameter_open(img_cc, rec_size, 8)        
        im2 = ccore.diameter_close(im1, int(rec_size / 2), 8)        

        #im1 = self.morpho_rec(img, rec_size)
        #im2 = self.morpho_rec2(im1, int(rec_size / 2))

        se = morphology.disk(se_size)        
        im3 = morphology.closing(im2.toArray(), se)
        
        return im3
    
    def h_minima(self, img, h):
        img_shift = img.copy()
        img_shift[img_shift >= 255 - h] = 255-h
        img_shift = img_shift + h
        rec = morphology.reconstruction(img_shift, img, method='erosion').astype(np.dtype('uint8'))
        diff = rec - img
        return diff
    
    def diameter_close(self, img, max_size):
        img_cc = ccore.numpy_to_image(img, copy=True)        
        res_cc = ccore.diameter_close(img_cc, max_size, 8)        
        res = res_cc.toArray()
        
        return res
    
    def diameter_tophat(self, img, max_size):
        img_cc = ccore.numpy_to_image(img, copy=True)        
        diam_close_cc = ccore.diameter_close(img_cc, max_size, 8)        
        diam_close = diam_close_cc.toArray()
        res = diam_close - img
        return res
    
    def split(self, img, imbin, alpha=0.5, dynval=2):
        pdb.set_trace()
        
        img_cc = ccore.numpy_to_image(img, copy=True)
        imbin_cc = ccore.numpy_to_image(imbin.astype(np.dtype('uint8')), copy=True)
        
        # inversion
        imbin_inv = ccore.linearRangeMapping(imbin_cc, 255, 0, 0, 255) 

        # distance function of the inverted image
        imDist = ccore.distanceTransform(imbin_inv, 2)
        
        # gradient of the image
        imGrad = ccore.externalGradient(img_cc, 1, 8)

        im1 = imDist.toArray()
        im2 = imGrad.toArray()
        im1 = im1.astype(np.dtype('float32'))
        im2 = im2.astype(np.dtype('float32'))
        
        temp = im1 + alpha * im2
        minval = temp.min()
        maxval = temp.max()
        
        if maxval==minval:
            return

        temp = 254 / (maxval - minval) * (temp - minval)
        temp = temp.astype(np.dtype('uint8'))
        temp_cc = ccore.numpy_to_image(temp, copy=True)
        
        ws = ccore.watershed_dynamic_split(temp_cc, dynval)
        res = ccore.infimum(ws, imbin_cc)
                
        return res
    
    
    def test_current(self, threshold1=4, threshold2=10):
    
        img = self.read_H_image()

        pref = self.prefilter(img, 20, 5)
        
        filename = os.path.join(test_out_folder, 'current_01_prefilter_%s.png' % self.image_name)
        skimage.io.imsave(filename, pref)

        diff1 = self.h_minima(pref, h=15)
        filename = os.path.join(test_out_folder, 'current_02_h_tophat_%s.png' % self.image_name)
        skimage.io.imsave(filename, 4*diff1)
        
        diff2 = self.diameter_tophat(pref, 80)
        filename = os.path.join(test_out_folder, 'current_03_diam_tophat_%s.png' % self.image_name)
        skimage.io.imsave(filename, diff2)

        res1 = np.zeros(diff1.shape)
        res1[diff1>threshold1] = 255
                
        res2 = np.zeros(diff2.shape)
        res2[diff2>threshold2] = 255
        
        overlay_img = self.overlay(img, res1, contour=True)
        filename = os.path.join(test_out_folder, 'current_04_overlay_h_tophat_%s.png' % self.image_name)
        skimage.io.imsave(filename, overlay_img)

        overlay_img = self.overlay(img, res2, contour=True)
        filename = os.path.join(test_out_folder, 'current_05_overlay_diamthresh_%s.png' % self.image_name)
        skimage.io.imsave(filename, overlay_img)
        
        res = res1
        res[res2>0] = 255
        overlay_img = self.overlay(img, res, contour=True)
        filename = os.path.join(test_out_folder, 'current_06_overlay_all_%s.png' % self.image_name)
        skimage.io.imsave(filename, overlay_img)

        filename = os.path.join(test_out_folder, 'current_07_original_%s.png' % self.image_name)
        skimage.io.imsave(filename, img)
        
        res_final = self.split(pref, res)
        
        # prefiltering removing bright structures inside the cells (opening by reconstruction)
        #pref = self.morpho_rec(img, 10)
        #filename = os.path.join(test_out_folder, 'current_01_%s.png' % self.image_name)
        #skimage.io.imsave(filename, pref)

        #res = self.difference_of_gaussian(pref, bigsize, smallsize)
        #filename = os.path.join(test_out_folder, 'current_02_%s_%i_%i.png' % (self.image_name, int(bigsize), int(smallsize)))
        #skimage.io.imsave(filename, res)
        return

    

if __name__ ==  "__main__":

    description =\
'''
%prog - running segmentation tool .
'''

    segmentation = Segmentation()
    segmentation.test_current()
    
    #segmentation.test_rough_detection()
    
    #segmentation.test_blob_detection()
    #segmentation.test_dog(30.0, 3.0, addon='30_3')
    #segmentation.test_dog(10.0, 1.0, addon='10_1')
    #segmentation.test_morpho2()
    #parser = OptionParser(usage="usage: %prog [options]",
    #                     description=description)
    
    #parser.add_option("-i", "--input_image", dest="input_image",
    #                  help="Input image")
    #parser.add_option("-o", "--output_folder", dest="output_folder",
    #                  help="Output folder")

    #(options, args) = parser.parse_args()

#     if (options.input_image is None) or (options.output_folder is None):
#         parser.error("incorrect number of arguments!")
# 
#     dec = Deconvolution()
#     dec(options.input_image, options.output_folder)

    print 'DONE'
    


    