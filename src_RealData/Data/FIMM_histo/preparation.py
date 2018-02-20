import os, sys

import numpy as np 

import vigra
from optparse import OptionParser

import pdb

def get_extensions(in_folder):

    max_width = 0
    max_height = 0    
    image_names = os.listdir(in_folder)
    image_names = sorted(filter(lambda x: os.path.splitext(x)[-1].lower() in ['.tif', '.tiff', '.png', '.jpg'], image_names))
    
    for image_name in image_names:
        img = vigra.readImage(os.path.join(in_folder, image_name))
        width = img.shape[0]
        height = img.shape[1]        
        print '%s: %i, %i' % (image_name, width, height)
        
        max_width = max(width, max_width)
        max_height = max(height, max_height)
        
    print 'maximal extensions: %i, %i' % (max_width, max_height)
    return max_width, max_height

def get_corner_color(colorin, width):    
    #avg_color = np.array([ np.mean(colorin[0:width,0:width,i]) for i in range(3)])
    avg_color = np.mean(np.mean(colorin[:width, :width], axis=1), axis=0)
    return avg_color
    
def adjust_images(in_folder, out_folder, max_width, max_height):
    image_names = os.listdir(in_folder)
    image_names = sorted(filter(lambda x: os.path.splitext(x)[-1].lower() in ['.tif', '.tiff', '.png', '.jpg'], image_names))
    
    ref_img = vigra.RGBImage((max_width, max_height))
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        print 'made %s' % out_folder
    for image_name in image_names:
        img = vigra.readImage(os.path.join(in_folder, image_name))
        width = img.shape[0]
        height = img.shape[1]        
        
        if width <= max_width and height <=max_height:        
            avg_color = get_corner_color(img, 5)
            ref_img[:,:,:] = avg_color
                    
            offset_x = (max_width - width) / 2
            offset_y = (max_height - height) / 2
            
            ref_img[offset_x:offset_x + width, offset_y:offset_y + height, :] = img        

        elif width > max_width and height > max_height:
            # in this case, we have a crop situation
            offset_x = (width - max_width) / 2
            offset_y = (height - max_height) / 2

            ref_img = img[offset_x:offset_x + max_width, 
                          offset_y:offset_y + max_height, :]
                       
            
        # export
        filename = os.path.join(out_folder, image_name)
        vigra.impex.writeImage(ref_img, filename)

    return



if __name__ ==  "__main__":

    description =\
'''
%prog - running segmentation tool .
'''

    parser = OptionParser(usage="usage: %prog [options]",
                         description=description)

    parser.add_option("-i", "--input_folder", dest="input_folder",
                      help="Input folder (raw data)")
    parser.add_option("-o", "--output_folder", dest="output_folder",
                      help="Output folder (properly adjusted images)")
    parser.add_option("--max_width", dest="max_width",
                      help="Maximal width (if not given, it is the taken as the max width of the images in the input folder")
    parser.add_option("--max_height", dest="max_height",
                      help="Maximal height (if not given, it is the taken as the max height of the images in the input folder")

    (options, args) = parser.parse_args()

    if (options.input_folder is None) or (options.output_folder is None):
        parser.error("incorrect number of arguments!")

    print
    print ' ******************* '
    print 'getting the maximal width and maximal height'
    if (options.max_width is None) or (options.max_height is None):
        max_width, max_height = get_extensions(options.input_folder)
    
    if not options.max_width is None: 
        max_width = int(options.max_width)

    if not options.max_height is None:
        max_height = int(options.max_height)
                
    print
    print ' ******************* '
    print 'adjusting the images'
    adjust_images(options.input_folder, options.output_folder, 
                  max_width, max_height)
    
    
    
    