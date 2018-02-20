#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import img_as_ubyte
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import FIMM_histo.deconvolution as deconv
from skimage import color
from scipy.misc import imsave

def flip_vertical(picture):
    """ 
    Vertical flip
    Takes an arbitrary image as entry
    """
    res = cv2.flip(picture, 1)
    return res


def flip_horizontal(picture):
    """
    Horizontal flip
    Takes an arbitrary image as entry
    """
    res = cv2.flip(picture, 0)
    return res


class Transf(object):
    """
    Generic python object for data augmentation
    Never call the transf class directly, always call
    one of it's child. He takes in entry array of images 
    and the generic function is _apply_.
    You can modify parameters, such as the interpolation
    method with SetFlag. The output type can be definied in 
    OutputType or you can overide it in the child.
    You can also override the name attribute.
    """
    def __init__(self, name):
        self.name = name

    def _apply_(self, *image):
        """
        Function to be called by DG object
        """
        raise NotImplementedError

    def enlarge(self, image, x, y):
        """
        Enlarges picture by mirror on border to assure that
        when  augmenting the picture we don't have holes.
        """
        rows, cols = image.shape[0], image.shape[1]
        if len(image.shape) == 2:
            enlarged_image = np.zeros(shape=(rows + 2 * y, cols + 2 * x))
        else:
            enlarged_image = np.zeros(shape=(rows + 2 * y, cols + 2 * x, 3))

        enlarged_image[y:(y + rows), x:(x + cols)] = image

        # top part:
        enlarged_image[0:y, x:(x + cols)] = flip_horizontal(
            enlarged_image[y:(2 * y), x:(x + cols)])

        # bottom part:
        enlarged_image[(y + rows):(2 * y + rows), x:(x + cols)] = flip_horizontal(
            enlarged_image[rows:(y + rows), x:(x + cols)])

        # left part:
        enlarged_image[y:(y + rows), 0:x] = flip_vertical(
            enlarged_image[y:(y + rows), x:(2 * x)])

        # right part:
        enlarged_image[y:(y + rows), (cols + x):(2 * x + cols)] = flip_vertical(
            enlarged_image[y:(y + rows), cols:(cols + x)])

        # top left from left part:
        enlarged_image[0:y, 0:x] = flip_horizontal(
            enlarged_image[y:(2 * y), 0:x])

        # top right from right part:
        enlarged_image[0:y, (x + cols):(2 * x + cols)] = flip_horizontal(
            enlarged_image[y:(2 * y), cols:(x + cols)])

        # bottom left from left part:
        enlarged_image[(y + rows):(2 * y + rows), 0:x] = flip_horizontal(
            enlarged_image[rows:(y + rows), 0:x])

        # bottom right from right part
        enlarged_image[(y + rows):(2 * y + rows), (x + cols):(2 * x + cols)] = flip_horizontal(
            enlarged_image[rows:(y + rows), (x + cols):(2 * x + cols)])
        enlarged_image = enlarged_image.astype('uint8')

        return(enlarged_image)

    def SetFlag(self, i):
        """
        Flag for the interpolation method to use.
        """
        if i == 0:
            return cv2.INTER_LINEAR
        elif i == 1:
            return cv2.INTER_NEAREST
        elif i == 2:
            return cv2.INTER_LINEAR

    def OutputType(self, image):
        """
        Force output type.
        """
        if image.dtype == "uint8":
            return image
        else:
            return image.astype(np.uint8)

class Identity(Transf):
    """
    As you would expect, it does nothing... A bit like you.
    Douch.
    """
    def __init__(self):

        Transf.__init__(self, "identity")

    def _apply_(self, *image):
        res = ()
        for img in image:
            res += (self.OutputType(img), )
        return image


class Translation(Transf):
    """
    Does a translation of vector (x,y) and takes as parameters x,y (INTEGER)
    """
    def __init__(self, x, y, enlarge=True):

        Transf.__init__(self, "Trans_" + str(x) + "_" + str(y))
        if x < 0:
            x = - x
            x_rev = -1
        else:
            x_rev = 1
        if y < 0:
            y = - y
            y_rev = -1
        else:
            y_rev = 1

        self.params = {"x": x, "y": y, "rev_x": x_rev,
                       "rev_y": y_rev, "enlarge": enlarge}

    def _apply_(self, *image):

        rows, cols, channels = image.shape

        x = self.params['x']
        y = self.params['y']
        rev_x = self.params['rev_x']
        rev_y = self.params['rev_y']
        enlarge = self.params['enlarge']
        res = ()
        n_img = 0
        for img in image:
            flags = self.SetFlag(n_img)
            if enlarge:
                big_image = self.enlarge(img, x, y)
                res += (big_image[(x + rev_x * x):(rows + x + rev_x * x),
                                  (y + rev_y * y):(cols + y + rev_y * y), :], )
            else:
                M = np.float32([[1, 0, (x * rev_x * -1)],
                                [0, 1, (y * rev_y * -1)]])
                res += (self.OutputType(cv2.warpAffine(image,
                                                       M, (cols, rows)), flags=flags), )
            n_img += 1
        return res


class Rotation(Transf):
    """
    Does a rotation of so much degrees.

    """
    def __init__(self, deg, enlarge=True):

        Transf.__init__(self, "Rot_" + str(deg))
        self.params = {"deg": deg, "enlarge": enlarge}

    def _apply_(self, *image):

        deg = self.params['deg']
        enlarge = self.params['enlarge']
        res = ()
        n_img = 0
        for img in image:
            rows, cols = img.shape[0], img.shape[1]
            flags = self.SetFlag(n_img)
            if enlarge:
                # this part could be better adjusted
                x = int(rows * (2 - 1.414213) / 1.414213)
                y = int(cols * (2 - 1.414213) / 1.414213)

                z = max(x, y)
                big_image = self.enlarge(img, z, z)

                b_rows, b_cols = big_image.shape[0], big_image.shape[1]
                M = cv2.getRotationMatrix2D((b_cols / 2, b_rows / 2), deg, 1)
                dst = cv2.warpAffine(big_image, M, (b_cols, b_rows), flags=flags)

                res += (self.OutputType(dst[z:(z + rows), z:(z + cols)]), )
            else:
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), deg, 1)
                sub_res = cv2.warpAffine(img, M, (cols, rows), flags=flags)
                res += (self.OutputType(sub_res),)
            n_img += 1

        return res


class Flip(Transf):
    """
    Does flips for 0 (vertical) and 1 (horizontal)
    """
    def __init__(self, hori):
        if hori != 0 and hori != 1:
            print "you must give a integer, your parameter is ignored"
            hori = 1
        Transf.__init__(self, "Flip_" + str(hori))
        self.params = {"hori": hori}

    def _apply_(self, *image):
        res = ()
        for img in image:
            hori = self.params["hori"]
            if hori == 1:
                sub_res = flip_horizontal(img)
            else:
                sub_res = flip_vertical(img)
            res += (self.OutputType(sub_res),)
        return res


class OutOfFocus(Transf):
    """
    Blurs the input images with a gaussian blurring of value sigma
    """
    def __init__(self, sigma):
        Transf.__init__(self, "OutOfFocus_" + str(sigma))
        self.params = {"sigma": sigma}

    def _apply_(self, *image):
        res = ()
        n_img = 0
        for img in image:
            if n_img == 1:
                sub_res = self.OutputType(img)
            else:
                sigma = self.params["sigma"]

                sub_res = cv2.blur(img, (sigma, sigma))

                sub_res = self.OutputType(sub_res)
            n_img += 1
            res += (sub_res,)
        return res


class ElasticDeformation(Transf):
    """
    Performs elastic deformation of the image in the same way.
    Several parameters:
    alpha: will be the scale by which we multiply the random displacement.
    sigma: will be the variance of the random displacement.
    Not to sure why sigma isn't directly incorporated in alpha... 
    (or vice-versa)
    alpha_affine: defines the bouding box for a variable I don't recall..


    """
    def __init__(self, alpha, sigma, alpha_affine, seed=None):
        Transf.__init__(self, "ElasticDeform_" + str(alpha) +
                        "_" + str(sigma) + "_" + str(alpha_affine))
        self.params = {"sigma": sigma,
                       "alpha": alpha, "alpha_affine": alpha_affine}

        self.seed = seed

    def grid(self, rows, cols, num_points):
        # returns a grid in the form of a stacked array x is 0 and y is 1
        src_cols = np.linspace(0, cols, num_points)
        src_rows = np.linspace(0, rows, num_points)
        src_rows, src_cols = np.meshgrid(src_rows, src_cols)
        src = np.dstack([src_cols.flat, src_rows.flat])[0]
        return src

    def _apply_(self, *image):
        res = ()
        n_img = 0
        for img in image:
            shape = img.shape
            shape_size = shape[:2]
            if not hasattr(self, "M"):
                alpha = img.shape[1] * self.params["alpha"]
                alpha_affine = img.shape[1] * self.params["alpha_affine"]
                sigma = img.shape[1] * self.params["sigma"]
                # Random affine
                center_square = np.float32(shape_size) // 2
                square_size = min(shape_size) // 3
                random_state = np.random.RandomState(None)

                pts1 = np.float32([center_square + square_size, [center_square[
                                  0] + square_size, center_square[1] - square_size], center_square - square_size])
                pts2 = pts1 + \
                    random_state.uniform(-alpha_affine, alpha_affine,
                                         size=pts1.shape).astype(np.float32)
                self.M = cv2.getAffineTransform(pts1, pts2)
                self.dx = gaussian_filter(
                    (random_state.rand(*shape) * 2 - 1), sigma) * alpha
                self.dy = gaussian_filter(
                    (random_state.rand(*shape) * 2 - 1), sigma) * alpha
            if shape[0:2] == self.dx.shape[0:2]:
                if len(shape) == 3:
                    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(
                        shape[0]), np.arange(shape[2]), indexing='ij')
                    indices = np.reshape(
                        x + self.dx, (-1, 1)), np.reshape(y + self.dy, (-1, 1)), np.reshape(z, (-1, 1))
                elif len(shape) == 2:
                    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), indexing='ij')
                    if len(self.dx.shape) == 3:
                        indices = np.reshape(x + np.mean(self.dx, axis=2), (-1, 1)), np.reshape(y + np.mean(self.dy, axis=2), (-1, 1))
                    else:
                        indices = np.reshape(x + self.dx, (-1, 1)), np.reshape(y + self.dy, (-1, 1))
                else:
                    print "Error"
            elif shape[0] > self.dx.shape[0]:
                print "not possible"
            else:
                offset = (self.dx.shape[0] - shape[0]) / 2
                if len(shape) == 3:
                    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(
                        shape[0]), np.arange(shape[2]), indexing='ij')
                    indices = np.reshape(
                        x + self.dx[offset:-offset,offset:-offset], (-1, 1)), np.reshape(y + self.dy[offset:-offset,offset:-offset], (-1, 1)), np.reshape(z, (-1, 1))
                elif len(shape) == 2:
                    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), indexing='ij')
                    if len(self.dx.shape) == 3:
                        indices = np.reshape(x + np.mean(self.dx[offset:-offset, offset:-offset], axis=2), (-1, 1)), np.reshape(y + np.mean(self.dy[offset:-offset, offset:-offset], axis=2), (-1, 1))
                    else:
                        indices = np.reshape(x + self.dx[offset:-offset, offset:-offset], (-1, 1)), np.reshape(y + self.dy[offset:-offset, offset:-offset], (-1, 1))
                else:
                    print "Error"
            if n_img == 1:
                order = 0
                flags = cv2.INTER_NEAREST
            else:
                order = 1
                flags = cv2.INTER_LINEAR
            img = cv2.warpAffine(img, self.M, shape_size[
                                 ::-1], flags=flags, borderMode=cv2.BORDER_REFLECT_101)

            sub_res = map_coordinates(
                img, indices, order=order, mode='reflect').reshape(shape)
            sub_res = self.OutputType(sub_res)
            res += (sub_res,)
            n_img += 1
        return res


def GreyValuePerturbation(image, k, b, MIN=0, MAX=255):
    """
    Performs an affine transformation of the greyscale image
    input:
        k: scale
        b: offset
        MIN and MAX (depends mostly on image type)
    """

    dims = image.shape
    if len(dims) != 2:
        raise ValueError('Wrong image dimension, it should be greyscale!')
    def AffineTransformation(x, aa=k, bb=b, nn=MIN, mm=MAX):
        if mm == 255:
            return max(nn, min(mm, int(aa * x + b)))
        else:
            return max(nn, min(mm, aa * x + b))

    f = np.vectorize(AffineTransformation)
    image = f(image)
    return image


class HE_Perturbation(Transf):
    """
    Transforms image in H/E, perfoms grey value variation on
    this subset and then transforms it back.
    WITH THOMAS RGB -> HE

    """
    def __init__(self, ch1, ch2, ch3 = (1,0)):
        k1, b1 = ch1
        k2, b2 = ch2
        k3, b3 = ch3
        Transf.__init__(self, "HE_Perturbation_Thomas_" + str(k1) +
                        "_" + str(b1) + "_" + str(k2) +
                        "_" + str(b2) + "_" + str(k3) +
                        "_" + str(b3) )
        k = [k1, k2, k3]
        b = [b1, b2, b3]
        self.params = {"k": k,
                       "b": b}
    def _apply_(self, *image):
        res = ()
        n_img = 0
        for img in image:
            if n_img == 0:
                ### transform image into HE
                dec = deconv.Deconvolution()
                dec.params['image_type'] = 'HEDab'

                np_img = np.array(img)
                dec_img = dec.colorDeconv(np_img)
                #pdb.set_trace()
                dec_img = dec_img.astype('uint8')
                ### perturbe each channel H, E, Dab
                for i in range(3):
                    k_i = self.params['k'][i]
                    b_i = self.params['b'][i]
                    val = np.max(dec_img[:,:,i])
                    dec_img[:,:,i] = GreyValuePerturbation(dec_img[:, :, i], k_i, b_i, 
                               MIN=0,
                               MAX=255)
                    val -= np.max(dec_img[:,:,i])
                    dec_img[:,:,i] += val
                sub_res = dec.colorDeconvHE(dec_img).astype('uint8')

                ### Have to implement deconvolution of the deconvolution


            else:
                sub_res = img

            res += (sub_res,)
            n_img += 1
        return res



class HE_Perturbation2(Transf):
    """
    Transforms image in H/E, perfoms grey value variation on
    this subset and then transforms it back. 1 is made with
    Thomas' rgb to he whereas this one is made with the one 
    from skimage.
    """
    def __init__(self, ch1, ch2, ch3 = (1,0)):
        k1, b1 = ch1
        k2, b2 = ch2
        k3, b3 = ch3
        Transf.__init__(self, "HE_Perturbation_" + str(k1) +
                        "_" + str(b1) + "_" + str(k2) +
                        "_" + str(b2) + "_" + str(k3) +
                        "_" + str(b3) )
        k = [k1, k2, k3]
        b = [b1, b2, b3]
        self.params = {"k": k,
                       "b": b}
    def _apply_(self, *image):
        res = ()
        n_img = 0
        for img in image:
            if n_img == 0:

                dec_img = color.rgb2hed(img)
                ### perturbe each channel H, E, Dab
                for i in range(3):
                    k_i = self.params['k'][i]
                    b_i = self.params['b'][i]
                    dec_img[:,:,i] = GreyValuePerturbation(dec_img[:, :, i], k_i, b_i)
                sub_res = color.hed2rgb(dec_img).astype('uint8')

                ### Have to implement deconvolution of the deconvolution


            else:
                sub_res = img

            res += (sub_res,)
            n_img += 1
        return res



class HSV_Perturbation(Transf):
    """
    Transforms image in H/E, perfoms grey value variation on
    this subset and then transforms it back.


    """
    def __init__(self, ch1, ch2, ch3 = (1,0)):
        k1, b1 = ch1
        k2, b2 = ch2
        k3, b3 = ch3
        Transf.__init__(self, "HSV_Perturbation_" + str(k1) +
                        "_" + str(b1) + "_" + str(k2) +
                        "_" + str(b2) + "_" + str(k3) +
                        "_" + str(b3) )
        k = [k1, k2, k3]
        b = [b1, b2, b3]
        self.params = {"k": k,
                       "b": b}
    def _apply_(self, *image):
        res = ()
        n_img = 0
        for img in image:
            if n_img == 0:
                #pdb.set_trace()
                ### transform image into HSV
                img = img_as_ubyte(color.rgb2hsv(img))
                ### perturbe each channel H, E, Dab
                for i in range(3):
                    k_i = self.params['k'][i] 
                    b_i = self.params['b'][i] 
                    img[:,:,i] = GreyValuePerturbation(img[:, :, i], k_i, b_i, MIN=0., MAX=255)
                    #plt.imshow(img[:,:,i], "gray")
                    #plt.show()
                sub_res = img_as_ubyte(color.hsv2rgb(img))
            else:
                sub_res = img

            res += (sub_res,)
            n_img += 1
        return res

def ListTransform(n_rot=4, n_elastic=50, n_he=50, n_hsv = 50,
                  var_elast=[1.2, 24. / 512, 0.07], var_hsv=[0.01, 0.07],
                  var_he=[0.07, 0.07]):
    """
    Returns a list of random tranformations for data augmentation.
    """
    transform_list = [Identity(), Flip(0), Flip(1)]
    if n_rot != 0:
        for rot in np.arange(1, 360, n_rot):
            transform_list.append(Rotation(rot, enlarge=True))

    for sig in [1, 2, 3, 4]:
        transform_list.append(OutOfFocus(sig))

    for i in range(n_elastic):
        transform_list.append(ElasticDeformation(var_elast[0], var_elast[1], var_elast[2]))

    k_h = np.random.normal(1.,var_he[0], n_he)
    k_e = np.random.normal(1.,var_he[1], n_he)

    for i in range(n_he):
        transform_list.append(HE_Perturbation((k_h[i],0), (k_e[i],0), (1, 0)))


    k_s = np.random.normal(1.,var_hsv[0], n_hsv)
    k_v = np.random.normal(1.,var_hsv[1], n_hsv)

    for i in range(n_hsv):
        transform_list.append(HSV_Perturbation((1,0), (k_s[i],0), (k_v[i], 0))) 

    transform_list_test = [Identity()]

    return transform_list, transform_list_test

