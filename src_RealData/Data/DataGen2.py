#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import numpy as np
from random import shuffle, randint, sample, seed
import random
import os
from scipy import misc
import nibabel as ni
import itertools
from skimage import measure
import pdb
import matplotlib.pylab as plt
import time
from joblib import Parallel, delayed
import FIMM_histo.deconvolution as deconv
from ImageTransform import flip_horizontal, flip_vertical
from utils import generate_wsl

def LoadImageBatch(path, img_format):
    """
    Generic function to be used in multi-threaded batch read for input images.
    """
    image = misc.imread(path)
    if image.shape[2] == 4:
        image = image[:, :, 0:3]

    if img_format == "HEDab":
        dec = deconv.Deconvolution()
        dec.params['image_type'] = 'HEDab'

        np_img = np.array(image)
        dec_img = dec.colorDeconv(np_img[:, :, :3])

        image = dec_img.astype('uint8')

    elif img_format == "HE":
        dec = deconv.Deconvolution()
        dec.params['image_type'] = 'HEDab'

        np_img = np.array(image)
        dec_img = dec.colorDeconv(np_img[:, :, :3])

        image = dec_img.astype('uint8')

    return image

def OpenReadProcess(img_p, lbl_p, DG, f, crop_n):
    """
    Generic function to be used in multi-threaded batch load.
    """
    img = LoadImageBatch(img_p, DG.img_format)
    lbl = LoadGTBatch(lbl_p)
    img_lbl = (img, lbl)
    img_lbl = DG.Process(img_lbl, f, crop_n)
    
    return img_lbl

def LoadGTBatch(path, normalize=True):
    """
    Generic function to be used in multi-threaded batch read for GT.
    """
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
    return img


class DataGen(object):
    """
    DataGen object at the core of the data handling.
    """
    def __init__(self, path, crop=1, size=None, transforms=None,
                 split="train", leave_out=1, seed_=None, name="optionnal",
                 img_format="RGB", UNet=False, N_JOBS=1, 
                 perc_trans=1., return_path=False, num="141549", 
                 mean_file=None):

        self.path = path
        self.name = name
        self.transforms = transforms
        self.perc_trans = perc_trans
        self.n_trans = int(self.perc_trans * len(self.transforms))
        self.crop = crop
        self.split = split
        self.leave_out = leave_out
        self.seed = seed_

        self.GetPatients(path)
        self.SortPatients()
        self.SetPatient(num)
        self.img_format = img_format

        
        self.return_path = return_path


        self.random_crop = True if size is not None else False
        self.size = size

        self.UNet = UNet
        self.N_JOBS = N_JOBS

        if mean_file is not None:
            self.Mean = True
            self.mean_array = np.load(mean_file)
        else:
            self.Mean = False

    def ReLoad(self, split):
        """
        This functions reset the inner variable of the object,
        it resets the path and finds added images.
        """
        self.split = split
        self.GetPatients(self.path)
        self.SortPatients()


    def __getitem__(self, key):
        """
        Returns for a given key
        key 0: patient
        key 1: image of patient
        key 2: transformation to apply
        key 3: crop number

        For examples, DG[1,3,2,0] will fetch from patient number 1,
        image number 3, it will apply transformation number 2 and return
        crop number 0.

        """

        if len(key) != 4:
            print "key given: ", key
            print "key length %d" % len(key)
            raise Exception('Wrong number of keys')

        if key[0] > len(self.patients_iter):
            raise Exception(
                "Value exceed number of patients available for {}ing.".format(self.split))
        numero = self.patients_iter[key[0]]
        n_patient = len(self.patient_img[numero])
        
        if key[1] > n_patient:
            raise Exception(
                "Patient {} doesn't have {} possible images.".format(self.patients_iter[key[0]], key[1]))
        
        if key[2] > self.n_trans:
            raise Exception(
                "Value exceed number of possible transformation for {}ing".format(self.split))

        if key[3] > self.crop - 1:
            raise Exception("Value exceed number of crops")

        img_path = self.patient_img[numero][key[1]]
        lbl_path = img_path.replace("Slide", "GT").replace(".png", ".nii.gz")
        
        img_lbl_Mwgt = ()
        img_lbl_Mwgt += (self.LoadImage(img_path), )
        img_lbl_Mwgt += (self.LoadGT(lbl_path), )


        f = self.transforms[key[2]]
        crop_n = key[3]
        img_lbl_Mwgt = self.Process(img_lbl_Mwgt, f, crop_n)

        if self.return_path:
            img_lbl_Mwgt += (img_path,)                    
        
        return img_lbl_Mwgt




    def KeyPath(self, key):
        """
        Checks if given key is possible and if so, returns path
        to raw image and GT.
        """
        if len(key) != 4:
            print "key given: ", key
            print "key length %d" % len(key)
            raise Exception('Wrong number of keys')

        if key[0] > len(self.patients_iter):
            raise Exception(
                "Value exceed number of patients available for {}ing.".format(self.split))
        numero = self.patients_iter[key[0]]
        n_patient = len(self.patient_img[numero])
        
        if key[1] > n_patient:
            raise Exception(
                "Patient {} doesn't have {} possible images.".format(self.patients_iter[key[0]], key[1]))
        
        if key[2] > self.n_trans:
            raise Exception(
                "Value exceed number of possible transformation for {}ing".format(self.split))

        if key[3] > self.crop - 1:
            raise Exception("Value exceed number of crops")

        img_path = self.patient_img[numero][key[1]]
        lbl_path = img_path.replace("Slide", "GT").replace(".png", ".nii.gz")
        return img_path, lbl_path

    def Process(self, img_lbl_Mwgt, f, crop_n):
        """
        Apply necessary transformation to input image and label.
        """

        if self.UNet:
            img_lbl_Mwgt = self.Unet_cut(*img_lbl_Mwgt)

        img_lbl_Mwgt = f._apply_(*img_lbl_Mwgt)  # change _apply_

        if self.crop != 1:
            img_lbl_Mwgt = self.DivideImage(crop_n, *img_lbl_Mwgt)

        if self.random_crop:
            img_lbl_Mwgt = self.CropImgLbl(*img_lbl_Mwgt)

        if self.UNet:
            CheckNumberForUnet(img_lbl_Mwgt[0].shape[0])
            img_lbl_Mwgt = self.ReduceDimUNet(*img_lbl_Mwgt)    

        if self.Mean:
            img_lbl_Mwgt = self.SubtractMean(*img_lbl_Mwgt)

        return img_lbl_Mwgt

    def LoadImage(self, path):
        """
        Loads one single input image with certain image format.
        This image format can be 'RGB', 'HEDab' and 'HE'.

        """

        if not hasattr(self, "img_format"):
            self.img_format = "RGB"

        image = misc.imread(path)
        if image.shape[2] == 4:
            image = image[:, :, 0:3]

        if self.img_format == "HEDab":
            dec = deconv.Deconvolution()
            dec.params['image_type'] = 'HEDab'

            np_img = np.array(image)
            dec_img = dec.colorDeconv(np_img[:, :, :3])

            image = dec_img.astype('uint8')

        elif self.img_format == "HE":
            dec = deconv.Deconvolution()
            dec.params['image_type'] = 'HEDab'

            np_img = np.array(image)
            dec_img = dec.colorDeconv(np_img[:, :, :3])

            image = dec_img.astype('uint8')

        return image


    def LoadGT(self, path, normalize=True):
        """
        Loads one single GT image.
        """
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
        return img


    def DivideImage(self, quarter, *iterable):
        """
        Crops image into number of crops, it takes an iterable in input
        in order to apply the same croping to each image of the iterable.
        """
        num_per_side = int(np.sqrt(self.crop))
        n_img = len(iterable)
        res = ()
        assert quarter  < self.crop, "Wrong slicing into crops... %d < %d" % (quarter, self.crop)

        x = iterable[1].shape[0]
        y = iterable[1].shape[1]

        if self.UNet:
            x -= 184
            y -= 184

        x_step = x / num_per_side
        y_step = y / num_per_side

        for n_img, iter in enumerate(iterable):

            i_old = 0
            number = 0
            for i in range(x_step, x + 1, x_step):
                j_old = 0
                for j in range(y_step, y + 1, y_step):
                    if number == quarter: 
                        
                        if self.UNet:
                            n = 92
                            res += (iter[(i_old):(i+2*n), (j_old):(j+2*n)],)
                        else:
                            res += (iter[i_old:i, j_old:j],)
                    j_old = j
                    number +=1

                i_old = i
        return res


    def CropImgLbl(self, *kargs):
        """
        Performs a random of crop on iterable of images.
        """
        dim = kargs[0].shape
        x = dim[0]
        y = dim[1]
        if self.UNet:
            x -= 184
            y -= 184

        x_prime = self.size[0]
        y_prime = self.size[1]
        x_rand = randint(0, x - x_prime)
        y_rand = randint(0, y - y_prime)
        res = ()
        for i in range(len(kargs)):

            if self.UNet:
                n = 92
                res += (self.RandomCropGen(kargs[i], (x_prime + 2*n, y_prime + 2*n), (x_rand, y_rand)),)
            else:
                res += (self.RandomCropGen(kargs[i], (x_prime, y_prime), (x_rand, y_rand)),)

        return res


    def RandomCropGen(self, img, size, shift):
        """
        Returns crop given crop parameters.
        """
        x_prime = size[0]
        y_prime = size[1]
        x_rand = shift[0]
        y_rand = shift[1]

        return img[x_rand:(x_rand + x_prime), y_rand:(y_rand + y_prime)]

    def ReduceDimUNet(self, *kargs):
        """
        If image has been augmented for the UNet processing. 
        Well this reduces it back. Only applies to first element of
        iterable.
        """
        res = ()
        res += (kargs[0],)
        n = 92
        for i in range(1, len(kargs)):
            res += (kargs[i][n:-n,n:-n],)
        return res

    def SubtractMean(self, *kargs):
        """
        Substracts means to first element of iterable if specified.
        """
        img = kargs[0]

        img = img.astype('float')
        img -= self.mean_array
        res = (img, )
        for i in range(1, len(kargs)):
            res += (kargs[i], )
        return res
        
    def Unet_cut(self, *kargs):
        """
        Applies UNet augmentation for borders to iterable.
        """
        res = ()
        for i in range(0, len(kargs)):
            res += (self.UNetAugment(kargs[i]),)
        return res

    def UNetAugment(self, img):
        """
        Performs augmentation of one image. It pads the mirror of the image
        to 92 pixels in each directions.
        """
        dim = img.shape
        i = 0
        new_dim = ()
        for c in dim:
            if i < 2:
                ne = c + 184
            else:
                ne = c
            i += 1
            new_dim += (ne, )

        result = np.zeros(shape=new_dim)
        n = 92
#       not right place to check for size..
#        assert CheckNumberForUnet(
#            dim[0] + 2 * n), "Dim not suited for UNet, it will create a wierd net"
        # middle
        result[n:-n, n:-n] = img.copy()
        # top middle
        result[0:n, n:-n] = flip_horizontal(result[n:(2 * n), n:-n])
        # bottom middle
        result[-n::, n:-n] = flip_horizontal(result[-(2 * n):-n, n:-n])
        # left whole
        result[:, 0:n] = flip_vertical(result[:, n:(2 * n)])
        # right whole
        result[:, -n::] = flip_vertical(result[:, -(2 * n):-n])
        result = result.astype("uint8")
        return result

    def RandomKey(self, rand):
        """
        Generates a random possible key.
        """
        if not rand:

            return [0] * 4

        else:
            a = randint(0, len(self.patients_iter) - 1)
            numero = self.patients_iter[a]
            b = randint(0, len(self.patient_img[numero]) - 1)
            c = randint(0, len(self.n_trans) - 1)
            d = randint(0, self.crop - 1)

            return [a, b, c, d]


    def NextKey(self, key):
        """
        Given a key, it generates the next possible key.
        """
        if key[3] == self.crop - 1:
            key[3] = 0  # crop
            if key[2] == len(self.transforms) - 1:
                key[2] = 0  # transform list
                numero = self.patients_iter[key[0]]
                if key[1] == len(self.patient_img[numero]) - 1:
                    key[1] = 0
                    if key[0] == len(self.patients_iter) - 1:
                        key[0] = 0
                        return key
                    else:
                        key[0] += 1
                        return key
                else:
                    key[1] += 1
                    return key
            else:
                key[2] += 1
                return key
        else:
            key[3] += 1
            return key


    def SetPatient(self, num):
        """
        Divides patients into test patients and train patients.
        """

        if isinstance(num, list):
            test_patient = num
        else:
            test_patient = [num]

        train_patient = [el for el in self.patient_num if el not in test_patient]

        if self.split == "train":

            images_train = [len(glob.glob(self.path + "/Slide_{}".format(el) + "/*.png")) for el in train_patient]
            self.length = np.sum(images_train) * self.crop * self.n_trans
            self.patients_iter = train_patient

        else:

            images_test = [len(glob.glob(self.path + "/Slide_{}".format(el) + "/*.png")) for el in test_patient]
            self.length = np.sum(images_test) * self.crop * self.n_trans
            self.patients_iter = test_patient

        self.SetRandomList() ## needed? 


    def GeneratePossibleKeys(self):
        """
        Exhaustive list of all possible keys.
        """
        len_key = 4

        AllPossibleKeys = []
        i = 0
        
        for num in self.patients_iter:
            lists = ([i],)
            i += 1
            nber_per_patient = len(self.patient_img[num])
            lists += (range(nber_per_patient),)
            lists += (np.random.randint(0, len(self.transforms), self.n_trans),)
            lists += (range(self.crop),)
            
            AllPossibleKeys += list(itertools.product(*lists))

        return AllPossibleKeys


    def SetRandomList(self):
        """
        Generates a random list of possible keys.
        """
        RandomList = self.GeneratePossibleKeys()
        shuffle(RandomList)
        self.RandomList = RandomList
        self.key_iter = 0


    def NextKeyRandList(self, key):
        """
        Returns next key in the random list of possible keys.
        """

        if not hasattr(self, "RandomList"):
            self.SetRandomList()
            self.key_iter = 0
        else:  
            self.key_iter += 1
        if self.key_iter == self.length:
            self.key_iter = 0

        return self.RandomList[self.key_iter]


    def GetPatients(self, path):
        """
        Fetchs images from patient folders.
        """

        folders = glob.glob(self.path + "/Slide_*")
        patient_num = []

        for el in folders:
            patient_num.append(el.split("_")[-1].split('.')[0])
        shuffle(patient_num)

        self.patient_num = patient_num
        self.patient_img = {el: glob.glob(self.path + "/Slide_{}".format(el) + "/*.png") for el in patient_num}


    def SortPatients(self):
        """
        Divives patients into train and test set.
        """
        if self.seed is not None:
            seed(self.seed)

        n = len(self.patient_num)
        test_patient = sample(self.patient_num, self.leave_out)
        train_patient = [el for el in self.patient_num if el not in test_patient]
        number_of_transforms = len(self.transforms)

        if self.split == "train":

            train_images = [len(glob.glob(self.path + "/Slide_{}".format(el) + "/*.png")) for el in train_patient]
            self.length = np.sum(train_images) * self.crop * self.n_trans
            self.patients_iter = train_patient

        else:
            test_images = [len(glob.glob(self.path + "/Slide_{}".format(el) + "/*.png")) for el in test_patient]
            self.length = np.sum(test_images) * self.crop * self.n_trans
            self.patients_iter = test_patient


    def SetTransformation(self, list_object):
        """
        Sets transform to given parameter.
        """

        self.transforms = list_object


    def SetPath(self, path):
        """
        Sets path.
        """
        self.path = path
        self.GetPatients(path)
        self.SortPatients()

    def Batch(self, key, batch_size):
        """ 
        Loading images in a multithreaded way to increase batch loads.
        """
        if self.UNet:
            Width_Images = self.size[0] + 184
            Height_Images = self.size[1] + 184
        else:
            Width_Images = self.size[0]
            Height_Images = self.size[1]

        ImagesBatch = np.zeros(shape=(batch_size, Width_Images, Height_Images, 3), dtype='float')
        LabelsBatch = np.zeros(shape=(batch_size, self.size[0], self.size[1], 1), dtype='float')

        MultipleProcess = self.N_JOBS != 1
        if not MultipleProcess:
            for i in range(batch_size):
                key = self.NextKeyRandList(0)
                ImagesBatch[i], LabelsBatch[i,:,:,0] = self[key]
        else:
            IMG_list, LBL_list = [], []
            f_list = []
            crop_list = []
            for i in range(batch_size):
                key = self.NextKeyRandList(0)
                img_path, lbl_path = self.KeyPath(key)
                IMG_list.append(img_path)
                LBL_list.append(lbl_path)
                f_list.append(self.transforms[key[2]])
                crop_list.append(key[3])

            ITERABLE = zip(IMG_list, LBL_list, f_list, crop_list)

            res = Parallel(n_jobs=self.N_JOBS)(delayed(OpenReadProcess)(img_p, lbl_p, self, f, crop_n) for img_p, lbl_p, f, crop_n in ITERABLE)

            for i in range(len(res)):
                ImagesBatch[i], LabelsBatch[i,:,:,0] = res[i]

        return ImagesBatch, LabelsBatch



def CheckNumberForUnet(n):
    """
    Checks if number is compatible with unet architecture.
    """
    ans = False
    if (n - 4) % 2 == 0:
        n = (n - 4) / 2
        if (n - 4) % 2 == 0:
            n = (n - 4) / 2
            if (n - 4) % 2 == 0:
                n = (n - 4) / 2
                if (n - 4) % 2 == 0:
                    n = (n - 4) / 2
                    ans = True

    return ans