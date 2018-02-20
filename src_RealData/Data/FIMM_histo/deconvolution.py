import numpy
from numpy import linalg
#import vigra

from optparse import OptionParser
import os
import sys
import numpy as np


class Deconvolution(object):

    def __init__(self):
        self.params = {
            'image_type': 'HEDab'
        }

        return

    def log_transform(self, colorin):
        res = - 255.0 / numpy.log(256.0) * numpy.log((colorin + 1) / 256.0)
        res[res < 0] = 0.0
        res[res > 255.0] = 255.0
        return res

    def exp_transform(self, colorin):
        res = numpy.exp((255 - colorin) * numpy.log(255) / 255)
        res[res < 0] = 0.0
        res[res > 255.0] = 255.0
        return res

    def colorDeconv(self, imin):
        M_h_e_dab_meas = numpy.array([[0.650, 0.072, 0.268],
                                      [0.704, 0.990, 0.570],
                                      [0.286, 0.105, 0.776]])

        # [H,E]
        M_h_e_meas = numpy.array([[0.644211, 0.092789],
                                  [0.716556, 0.954111],
                                  [0.266844, 0.283111]])

        if self.params['image_type'] == "HE":
            # print "HE stain"
            M = M_h_e_meas
            M_inv = numpy.dot(linalg.inv(numpy.dot(M.T, M)), M.T)

        elif self.params['image_type'] == "HEDab":
            # print "HEDab stain"
            M = M_h_e_dab_meas
            M_inv = linalg.inv(M)

        else:
            # print "Unrecognized image type !! image type set to \"HE\" "
            M = numpy.diag([1, 1, 1])
            M_inv = numpy.diag([1, 1, 1])

        imDecv = numpy.dot(self.log_transform(imin.astype('float')), M_inv.T)
        imout = self.exp_transform(imDecv)

        return imout
        
    def colorDeconvHE(self, imin):
        """
        Does the opposite of colorDeconv
        """
        M_h_e_dab_meas = numpy.array([[0.650, 0.072, 0.268],
                                      [0.704, 0.990, 0.570],
                                      [0.286, 0.105, 0.776]])

        # [H,E]
        M_h_e_meas = numpy.array([[0.644211, 0.092789],
                                  [0.716556, 0.954111],
                                  [0.266844, 0.283111]])

        if self.params['image_type'] == "HE":
            # print "HE stain"
            M = M_h_e_meas
            
        elif self.params['image_type'] == "HEDab":
            # print "HEDab stain"
            M = M_h_e_dab_meas

        else:
            # print "Unrecognized image type !! image type set to \"HE\" "
            M = numpy.diag([1, 1, 1])
            M_inv = numpy.diag([1, 1, 1])

        imDecv = numpy.dot(self.log_transform(imin.astype('float')), M.T)
        imout = self.exp_transform(imDecv)
#        imout = numpy.zeros(imDecv.shape, dtype = numpy.uint8)

        # Normalization
#         for i in range(imout.shape[-1]):
#             toto = imDecv[:,:,i]
#             vmax = toto.max()
#             vmin = toto.min()
#             if (vmax - vmin) < 0.0001:
#                 continue
#             titi = (toto - vmin) / (vmax - vmin) * 255
#             titi = titi.astype(numpy.uint8)
#             imout[:,:,i] = titi

        return imout


# DISABLED VIGRA AS IT IS NOT INSTALLED ON KEPLER


#     def __call__(self, filename, out_path):
#         if not os.path.exists(out_path):
#             os.makedirs(out_path)
#             print 'made %s' % out_path

#         colorin = vigra.readImage(filename)
#         filename_base, extension = os.path.splitext(os.path.basename(filename))

#         col_dec = self.colorDeconv(colorin)
#         channels = ['h', 'e', 'dab']
#         for i in range(3):
#             new_filename = os.path.join(out_path,
#                                         filename_base + '__%s' % channels[i] + extension)
#             vigra.impex.writeImage(col_dec[:,:,i], new_filename)
#             print 'written %s' % new_filename

#         return

#     def generate_dec_crops(self, filename, out_path, crop_size=1024, nb_positions=1):
#         if not os.path.exists(out_path):
#             os.makedirs(out_path)
#             print 'made %s' % out_path

#         colorin = vigra.readImage(filename)
#         width = colorin.shape[0]
#         height = colorin.shape[1]

#         frow = np.sqrt(nb_positions)
#         if np.abs(int(frow) - frow) > 1e-10:
#             raise ValueError('number of positions needs to be squared.')
#         frow = int(frow)

#         if frow*crop_size > width or frow*crop_size > height:
#             print 'crop_size is too large (exceeds image dimensions) ... skipping %s' % filename
#             return

#         offset_x = (width - frow*crop_size) / 2
#         offset_y = (height - frow*crop_size) / 2

#         filename_base, extension = os.path.splitext(os.path.basename(filename))

#         col_dec = self.colorDeconv(colorin)
#         channels = ['h', 'e', 'dab']

#         for i in range(3):

#             position = 1
#             for y in range(frow):
#                 for x in range(frow):

#                     ref_img = col_dec[offset_x+x*crop_size:offset_x+(x+1)*crop_size,
# offset_y+y*crop_size:offset_y+(y+1)*crop_size, i]

#                     new_filename = os.path.join(out_path,
#                                                 '%s__P%05i__%s%s' % (filename_base, position, channels[i], extension))
#                     vigra.impex.writeImage(ref_img, new_filename)
#                     print 'written %s' % new_filename
#                     position += 1

#         return

#     def process_folder(self, input_folder, output_folder, crop_size, nb_positions):
#         image_names = filter(lambda x: os.path.splitext(x)[-1].lower() in ['.tiff', '.png', '.tif'], os.listdir(input_folder))
#         for image_name in image_names:
#             full_filename = os.path.join(input_folder, image_name)
#             self.generate_dec_crops(full_filename, output_folder, crop_size, nb_positions)

#         return


# if __name__ ==  "__main__":

#     description =\
# '''
# %prog - running segmentation tool .
# '''

#     parser = OptionParser(usage="usage: %prog [options]",
#                          description=description)

#     parser.add_option("-i", "--input_folder", dest="input_folder",
#                       help="Input folder")
#     parser.add_option("-o", "--output_folder", dest="output_folder",
#                       help="Output folder")
#     parser.add_option("--crop", dest="crop",
#                       help="Crop size")
#     parser.add_option("--fields", dest="fields",
# help="number of fields to generate (there will be <fields> fields of
# size <crop>, centered in the image")


#     (options, args) = parser.parse_args()

#     if (options.input_folder is None) or (options.output_folder is None):
#         parser.error("incorrect number of arguments!")

#     if options.crop is None:
#         crop_size = 1024
#     else:
#         crop_size = int(options.crop)

#     if options.fields is None:
#         fields = 1
#     else:
#         fields = int(options.fields)

#     dec = Deconvolution()
#     dec.process_folder(options.input_folder, options.output_folder, crop_size, fields)
#     print 'DONE'
