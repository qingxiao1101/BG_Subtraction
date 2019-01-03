#!/usr/bin/env python3
import math
from skimage import io
import numpy as np 
from time import time
from numba import autojit, jit, cuda
import cv2
import scipy.io as sio 
from pandas import DataFrame
from utils_func import dist_L1_batch, dist_hamming_batch 
from utils_func import normalized_min_dist_and_match
from utils_func import compute_one_pixel_gpu_kernel
import utils_func
#import lbsp
import configuration as config
from pyculib import rand as curand

def timer(func):
    def deco(*args, **kwargs):  
        start = time()
        res = func(*args, **kwargs)
        stop = time()
        print('function (%s) cost %f seconds' %(func.__name__,stop-start))
        return res 
    return deco

class Model():

    def __init__(self,init_image,init_images):

        self.image_path = 'image_path'
        self.current_frame = 1
        img_shape = init_image.shape
        if len(img_shape) == 3: 
            self.img_height,self.img_width,self.img_channel = img_shape
        else:
            self.img_height,self.img_width = img_shape
            self.img_channel = 1

        self.bg_samples = np.zeros(
            (self.img_height,self.img_width,config.number_samples,config.attribute),dtype = np.uint8)
        self.bg_weights = np.zeros(
            (self.img_height,self.img_width,config.number_samples),dtype = np.uint16)
        # Dmin long term and short term of 2D array
        self.weber_min_lt_array = np.zeros((self.img_height,self.img_width))
        self.weber_min_st_array = np.zeros((self.img_height,self.img_width))
        # weber_rate
        #self.weber_rate_lt_array = np.zeros((self.img_height,self.img_width),dtype=float)
        # last mask of 2D array, either forground or background, if bg, then true
        self.last_mask = np.full((self.img_height,self.img_width),True)

        # last rgb values, scaled in range (0,255)
        if self.img_channel==3:
            self.last_rgb_values = np.zeros((self.img_height,self.img_width,3),dtype=np.uint8)
        else:
            self.last_rgb_values = np.zeros((self.img_height,self.img_width),dtype=np.uint8)
        # threshold Rx
        self.model_Rx = np.ones((self.img_height,self.img_width))
        # update rate Tx
        self.model_Tx = np.full((self.img_height,self.img_width),1.0)
        # auxiliary parameter
        self.model_Vx = np.zeros((self.img_height,self.img_width))
        # auxiliary parameter
        self.model_blink_frequence = np.ones((self.img_height,self.img_width))

        self.min_weight_index = np.zeros((self.img_height,self.img_width),dtype=np.uint8)
        self.max_weight = np.zeros((self.img_height,self.img_width))

        self.mask = np.full((self.img_height,self.img_width),False)
        self.motion_region = np.full((self.img_height,self.img_width),False)
        self.min_weber_rate = np.zeros((self.img_height,self.img_width))
        # block and grid`s configuration of GPU
        if config.using_gpu:  
            self.BLOCKDIM = (16,8)
            blockspergrid_w = math.ceil(self.img_width/self.BLOCKDIM[0])
            blockspergrid_h = math.ceil(self.img_height/self.BLOCKDIM[1])
            self.GRIDDIM = (blockspergrid_w,blockspergrid_h)
        if self.img_channel==3:
            self.__init_bg_samples(init_image,init_images)
        # debug data
        self.debug_data = np.zeros((self.img_height,self.img_width))

    def __init_bg_samples(self,init_image,init_images):



        for i in range(config.number_samples):
            self.bg_samples[:,:,i,0] = init_image[:,:,0].copy()
            self.bg_samples[:,:,i,1] = init_image[:,:,1].copy()
            self.bg_samples[:,:,i,2] = init_image[:,:,2].copy()
            self.bg_samples[:,:,i,3] = np.full((self.img_height,self.img_width),20)
            self.bg_weights[:,:,i] = np.full((self.img_height,self.img_width),2000)

        num_of_image = init_images.shape[-1]

        for w in range(self.img_width):
            for h in range(self.img_height):
                for idx in range(8):
                    y,x = utils_func.get_neighboor_coordinate(h,w,self.img_width,self.img_height,idx)
                    self.bg_samples[h,w,idx,0] = init_image[y,x,0]
                    self.bg_samples[h,w,idx,1] = init_image[y,x,1]
                    self.bg_samples[h,w,idx,2] = init_image[y,x,2]
        
        for i in range(8,8+num_of_image):
            self.bg_samples[:,:,i,0] = init_images[:,:,0,i-8].copy()
            self.bg_samples[:,:,i,1] = init_images[:,:,1,i-8].copy()
            self.bg_samples[:,:,i,2] = init_images[:,:,2,i-8].copy()
        
    #@timer                
    def __matching_BG(self,current_rgb_values):
        if config.using_gpu:
            samples = np.ascontiguousarray(self.bg_samples)
            weights = np.ascontiguousarray(self.bg_weights)
            #temp = np.maximum(np.ones((self.img_height,self.img_width)),self.model_Tx.copy())
            if_update_samples = \
                utils_func.pixel_updated_probability(self.model_Tx.copy().astype(np.int32))
            if_update_neighboors = \
                utils_func.pixel_updated_probability(self.model_Tx.copy().astype(np.int32))
            stochastic = np.random.rand(self.img_height,self.img_width)
            random_neighboor_idx = np.random.randint(0,8,(self.img_height,self.img_width))
            which_samples = utils_func.random_choose_sample(np.full(self.mask.shape,config.number_samples))
            compute_one_pixel_gpu_kernel[self.GRIDDIM,self.BLOCKDIM](
                self.mask,
                self.min_weber_rate,
                self.max_weight,
                self.min_weight_index,
                self.model_Rx,
                if_update_samples,
                if_update_neighboors,
                samples,
                weights,
                current_rgb_values,
                stochastic,
                random_neighboor_idx,
                which_samples)
            #self.__update_samples(current_rgb_values)
            if config.if_stochastic_update and config.if_update_neighboor == True:
                pass
                #self.__update_samples(current_rgb_values)
                #mask = np.logical_and(if_update_samples,if_update_neighboors)
                #which_samples = np.where(mask==True,which_samples,np.full(which_samples.shape,-1))
                #self.__update_neighboor_random(self.bg_samples,which_samples,current_rgb_values)
        else:
            pass

        self.weber_min_st_array = self.weber_min_st_array*(1 - config.learning_rate_st) + \
                                    self.min_weber_rate*config.learning_rate_st

        self.weber_min_lt_array = self.weber_min_lt_array*(1 - config.learning_rate_lt) + \
                                    self.min_weber_rate*config.learning_rate_lt


    def __choose_updated_samples(self,mask:np.array,time_subsamples:np.array)->np.array:
        height,width = mask.shape

        if_updates = utils_func.pixel_updated_probability(time_subsamples.astype(np.int32))
        if_updates = np.where(mask==True,if_updates,False)
        which_samples = utils_func.random_choose_sample(np.full(mask.shape,config.number_samples))
        which_samples = np.where(if_updates==True,which_samples,-1)
        return which_samples

    @jit
    def __update_neighboor_random(self,bg_samples,which_sample,current_image_rgb):
        if which_sample >= 0:
            for w in range(self.img_width):
                for h in range(self.img_height):
                    y,x = utils_func.random_choose_neighboor(h,w,self.img_width,self.img_height,int(self.model_Tx[h,w].copy()))
                    if w!=x or h!=y :
                        bg_samples[y,x,which_sample,0] = current_image_rgb[h,w,0]
                        bg_samples[y,x,which_sample,1] = current_image_rgb[h,w,1]
                        bg_samples[y,x,which_sample,2] = current_image_rgb[h,w,2]

    @autojit
    def __update_samples_and_neighboor(self,
                                        bg_samples,
                                        which_samples:np.array,
                                        time_subsamples:np.array,
                                        current_image_rgb:np.array):
        height,width = which_samples.shape
        for w in range(width):
            for h in range(height):
                which_sample = which_samples[h,w]
                if which_sample >= 0:
                    bg_samples[h,w,which_sample,0] = current_image_rgb[h,w,0]
                    bg_samples[h,w,which_sample,1] = current_image_rgb[h,w,1]
                    bg_samples[h,w,which_sample,2] = current_image_rgb[h,w,2]
                    
                    y,x = utils_func.random_choose_neighboor(h,w,width,height,int(time_subsamples[h,w]))
                    if w!=x or h!=y :
                        bg_samples[y,x,which_sample,0] = current_image_rgb[h,w,0]
                        bg_samples[y,x,which_sample,1] = current_image_rgb[h,w,1]
                        bg_samples[y,x,which_sample,2] = current_image_rgb[h,w,2]
                    
    def __update_samples(self, current_image_rgb):
        which_samples = self.__choose_updated_samples(self.mask,self.model_Tx)
        self.__update_samples_and_neighboor(self.bg_samples,
                                            which_samples,
                                            self.model_Tx,
                                            current_image_rgb)

    def __new_compute_blink_frequence(self):
        blinks = self.mask ^ self.last_mask

        self.model_blink_frequence = \
            (self.current_frame*self.model_blink_frequence + blinks) / (self.current_frame+1)
        self.model_blink_frequence = (self.model_blink_frequence) / (np.max(self.model_blink_frequence)+0.001)

    def __compute_parameter_Vx(self):
        blinks = self.mask ^ self.last_mask

        self.model_Vx = self.model_Vx*100.0
        self.model_Vx += np.where(blinks==True,config.Vx_INCR,config.Vx_DECR)

        self.model_Vx = np.where(self.model_Vx<1,1.0,self.model_Vx) 
        self.model_Vx = np.where(self.model_Vx>100,100.0,self.model_Vx)
        self.model_Vx = self.model_Vx/100.0

    def __compute_parameter_Rx(self):
        
        weber_min = utils_func.batch_min(self.weber_min_st_array,self.weber_min_lt_array)
        #self.debug_data = weber_min
        mask = np.where(self.model_Rx<(1+weber_min*2)**2,True,False)
        #mask = np.where(weber_min>0.0,True,False)
        #self.save_variables(weber_min)
        self.model_Rx += \
            np.where(mask==True,0.05*(np.square(self.model_Vx-0.1)) ,-0.001/(self.model_Vx))
            #np.where(mask==True,0.2*self.model_blink_frequence**2 ,-0.001/(self.model_blink_frequence ))
        self.model_Rx = \
            np.where(self.model_Rx<config.model_Rx_lower,config.model_Rx_lower,self.model_Rx)
        self.model_Rx = \
            np.where(self.model_Rx>config.model_Rx_upper,config.model_Rx_upper,self.model_Rx)
        #self.save_variables((self.model_Rx-1)/4)
     
    def __compute_parameter_Tx(self):
        max_Dmin = utils_func.batch_max(self.weber_min_lt_array,self.weber_min_st_array)
        min_Dmin = utils_func.batch_min(self.weber_min_lt_array,self.weber_min_st_array)
        multi = min_Dmin*self.model_Vx
        multi = np.where(multi<0.001,0.001,multi)
        max_Dmin = np.where(max_Dmin<0.001,0.001,max_Dmin)
        temp = np.where(self.mask==False,1/multi,0.0)
        #self.save_variables(temp/np.max(temp))
        dev = (self.model_Vx)/max_Dmin
        temp_dev = np.where(self.mask==True,dev,0.0)
        #self.model_Tx += np.where(self.mask==False,0.05/multi,-0.5*dev) #for dynamic bg
        self.model_Tx += np.where(self.mask==False,0.1/multi,-100*dev) #for statistic bg
        #self.model_Tx -= 0.5*dev
        self.model_Tx = np.where(self.model_Tx<config.Tx_lower,config.Tx_lower,self.model_Tx)
        self.model_Tx = np.where(self.model_Tx>config.Tx_upper,config.Tx_upper,self.model_Tx)
        #self.save_variables(self.model_Tx/255)
        #self.save_variables(temp_dev/np.max(temp_dev))

    def __update_parameter(self,mask,rgb_values):

        self.__compute_parameter_Vx()
        self.__new_compute_blink_frequence()
        self.__compute_parameter_Rx()
        self.__compute_parameter_Tx()
        self.motion_region = np.where(self.model_blink_frequence>0.2,True,False)
        self.last_mask = mask.copy()
        self.last_rgb_values = rgb_values.copy()

    def __post_processing(self,image):
        kernel = np.ones((5, 5), np.uint8)
        img_medianBlur=cv2.medianBlur(image,9)
        dilation = cv2.dilate(img_medianBlur, kernel)
        #dilation = cv2.dilate(dilation, kernel)
        #dilation = cv2.dilate(dilation, kernel)
        return dilation

    def lighting_check(self,current_image):
        a = 0.7
        max_rgb = np.max(current_image,axis=2)/255.0
        min_rgb = np.min(current_image,axis=2)/255.0
        C = max_rgb - min_rgb
        V = max_rgb.copy()
        L = np.where(V<=a,V*(1/a),(1-V)*(1/(1-a)))
        S = np.minimum(C/L,np.full((self.img_height,self.img_width),1.0))
        lighting_mask = np.where(S<0.4,True,False)
        return lighting_mask

    def segmentation_image(self,mask,current_image)->np.array:
        img_shape = mask.shape
        #mask_image = is_matched_BG_samples_compile_optimal(current_image)
        seg_image = np.zeros((img_shape[0],img_shape[1]),dtype = np.uint8)
        forground = np.full((img_shape[0],img_shape[1]),255,dtype = np.uint8)

        seg_image = np.where(mask==False,forground,0)
        seg_image = np.where(self.motion_region==True,seg_image,0)
        #lighting_region = self.lighting_check(current_image)
        #seg_image = np.where(lighting_region==True,0,seg_image)

        return seg_image

    #@timer
    def ones_iteration(self,current_image,image_path):
        self.image_path = image_path
        if self.img_channel==3:
            self.__matching_BG(current_image)
            seg_image = self.segmentation_image(self.mask,current_image)
            #seg_image = self.__post_processing(seg_image)
            self.__update_parameter(self.mask,current_image)
            self.current_frame += 1
        else:
            pass
        return seg_image

    def save_variables(self,binary_path,image):
        #save_path = os.path.join('./dmin_image/', image_path.split('/')[-1])
        #saved_images = config.save_path + str(self.current_frame) + '.jpg'
        saved_images = binary_path + "bin" + self.image_path.split('/')[-1][2:]
        io.imsave(saved_images,image)
    def save_mat(self,data):
        sio.savemat('saveddata.mat', {'data': data})


    def debug_one_pixel_print(self,h=0,w=0):
        frame = {"R":self.bg_samples[h,w,:,0],
                    "G":self.bg_samples[h,w,:,1],
                    "B":self.bg_samples[h,w,:,2],
                    "Weight1":self.bg_samples[h,w,:,3],
                    "Weight2":self.bg_weights[h,w,:]}
        cell = DataFrame(frame,index=list(range(0,config.number_samples)))
        print(cell)



