#!/usr/bin/env python3
import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from skimage import io, transform
from pandas import Series,DataFrame
import pickle
from dist_utils import Distance
import lbsp

file_path = './test_input/'

NUM_SAMPLE = 50
ATTRIBUTE = 2
RGB_THRESHOLD = 30


def rgb2gray(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


class Model():

    def __init__(self,using_gpu = True):   #constructor
        self.__init_global_variant(using_gpu)
        self.__init_bg_samples()
        

    def __read_image_paths(self, filePath = file_path )->[str]: 
        img_names = [x for x in os.listdir(filePath) if x.split('.')[-1] in 'jpg|png']
        img_names = sorted(img_names, key = lambda x: int(x.split('.')[-2][2:]))
        img_paths = [os.path.join(filePath, x) for x in img_names]
        return img_paths

    def read_one_image(self, imgPath ) -> np.array: #return uint8
        img = cv2.imread(imgPath)
        return img

    def __init_global_variant(self,using_gpu):

        self.img_paths = self.__read_image_paths()
        h,w,c = self.read_one_image(self.img_paths[0]).shape
        self.img_width = w
        self.img_height = h
        self.img_channel = c
        if using_gpu :
            self.compute_one_image_lbsp_value = lbsp.lbsp_image_gpu
        else:
            self.compute_one_image_lbsp_value = lbsp.lbsp_image_normal

    '''
        there are 2 kinds of initialization: 1. using single frame and 2. using 50 frames to initial bg_samples
        here choose 1. 
    '''
    def __init_bg_samples(self):
        self.bg_samples = np.zeros((self.img_height,self.img_width,NUM_SAMPLE,ATTRIBUTE),dtype = int)

        first_image = self.read_one_image(self.img_paths[0])
        init_lbsp = np.zeros((first_image.shape[0],first_image.shape[1]))
        img_gray = cv2.cvtColor(first_image,cv2.COLOR_BGR2GRAY)
        init_lbsp = np.zeros(img_gray.shape,dtype=np.uint16)
        rgb_values = np.zeros(img_gray.shape,dtype=int)

        self.compute_one_image_lbsp_value(img_gray,init_lbsp)
        rgb_values = self.compute_RGB_values(first_image)
        for i in range(NUM_SAMPLE):
            self.bg_samples[:,:,i,0] = rgb_values.copy()
            self.bg_samples[:,:,i,1] = init_lbsp.copy()


    def debug_one_pixel_print(self,h=0,w=0):
        frame = {"RGB":self.bg_samples[h,w,:,0],
                 "LBSP":self.bg_samples[h,w,:,1]}
        cell = DataFrame(frame,index=list(range(1,NUM_SAMPLE+1)))
        print(cell)  

    def is_matched_background_samples(self,current_image):
        rgb_values = self.bg_samples[:,:,0,0] 
        mark_rgb_match = np.full(rgb_values.shape,False)
        mark_rgb_match = np.where(Distance.dist_L1_batch(rgb_values,self.compute_RGB_values(current_image)) <= RGB_THRESHOLD,True,False)
        return mark_rgb_match

    def update_samples(self,current_image):
        pass
    def segmentation_image(self,current_image):
        pass

    def compute_RGB_values(self,current_image):
        img_shape = current_image.shape
        rgb_values = np.zeros((current_image.shape[0],current_image.shape[1]))
        if len(img_shape) == 3: 
            rgb_values = np.sum(current_image,axis=2)
        elif len(img_shape) == 2:
            rgb_values = current_image.copy()
        else:
            raise TypeError("shape of input image error.")
        return rgb_values



if __name__ == '__main__':

    data = Model()
    image = data.read_one_image(data.img_paths[100])
    test = data.is_matched_background_samples(rgb2gray(image))
    print(test[200:210,400:410])


