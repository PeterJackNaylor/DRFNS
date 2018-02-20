#!/usr/bin/env python
# -*- coding: utf-8 -*-

from DataGen2 import DataGen
import glob
import itertools
from random import shuffle, sample, seed
import numpy as np


class DataGenRandomT(DataGen):
    """
    Other DG object with better random key generators.
    """
    def SetPatient(self, num):
        """
        Set and sorts patients for train and test.
        """
        if isinstance(num, list):
            test_patient = num
        else:
            test_patient = [num]

        train_patient = [el for el in self.patient_num if el not in test_patient]

        if self.split == "train":

            images_train = [len(glob.glob(self.path + "/Slide_{}".format(el) + "/*.png")) for el in train_patient]
            self.length = np.sum(images_train) * self.crop
            self.patients_iter = train_patient

        else:

            images_test = [len(glob.glob(self.path + "/Slide_{}".format(el) + "/*.png")) for el in test_patient]
            self.length = np.sum(images_test) * self.crop 
            self.patients_iter = test_patient

        self.SetRandomList() 


    def GeneratePossibleKeys(self):
        """
        Generates all possible keys.
        """
        len_key = 4

        AllPossibleKeys = []
        i = 0
        
        for num in self.patients_iter:
            lists = ([i],)
            i += 1
            nber_per_patient = len(self.patient_img[num])
            lists += (range(nber_per_patient),)
            lists += ([-1],)
            lists += (range(self.crop),)
            
            AllPossibleKeys += list(itertools.product(*lists))

        return AllPossibleKeys


    def SetRandomList(self):
        """
        Sets random list
        """

        RandomList = self.GeneratePossibleKeys()
        shuffle(RandomList)
        self.RandomList = RandomList
        self.key_iter = 0


    def NextKeyRandList(self, key):
        """
        Returns next key in random list.
        """

        if not hasattr(self, "RandomList"):
            self.SetRandomList()
            self.key_iter = 0
        else:  
            self.key_iter += 1
        if self.key_iter == self.length:
            self.key_iter = 0

        a, b, c, d = self.RandomList[self.key_iter]
        c = np.random.randint(0, self.n_trans)
        

        return (a, b, c, d)

    def SortPatients(self):
        """
        Sorts patients for train and test set.
        """
        if self.seed is not None:
            seed(self.seed)

        n = len(self.patient_num)
        test_patient = sample(self.patient_num, self.leave_out)
        train_patient = [el for el in self.patient_num if el not in test_patient]
        number_of_transforms = len(self.transforms)

        if self.split == "train":

            train_images = [len(glob.glob(self.path + "/Slide_{}".format(el) + "/*.png")) for el in train_patient]
            self.length = np.sum(train_images) * self.crop
            self.patients_iter = train_patient

        else:
            test_images = [len(glob.glob(self.path + "/Slide_{}".format(el) + "/*.png")) for el in test_patient]
            self.length = np.sum(test_images) * self.crop
            self.patients_iter = test_patient

