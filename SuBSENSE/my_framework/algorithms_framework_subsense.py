#!/usr/bin/env python3
import math
import numpy as np 
from time import time
from numba import autojit, jit, cuda

from utils_func import dist_L1_batch, dist_hamming_batch 
from utils_func import normalized_min_dist_and_match
from utils_func import normalized_min_dist_and_match_gpu_kernel
import utils_func
import lbsp
import configuration as config


def timer(func):
    def deco(*args, **kwargs):  
        start = time()
        res = func(*args, **kwargs)
        stop = time()
        print('function (%s) cost %f seconds' %(func.__name__,stop-start))
        return res 
    return deco

class Model():

    def __init__(self,init_image):
        '''
        self.number_samples = number_samples
        self.attribute = attribute
        self.rgb_init_threshold = rgb_init_threshold
        self.lbsp_init_threshold = lbsp_init_threshold
        self.match_threshold = match_threshold
        self.learning_rate_st = learning_rate_st
        self.learning_rate_lt = learning_rate_lt
        self.using_gpu = using_gpu
        '''

        self.current_frame = 0
        img_shape= init_image.shape
        if len(img_shape) == 3: 
            self.img_height,self.img_width,self.img_channel = img_shape
        else:
            self.img_height,self.img_width = img_shape
            self.img_channel = 1

        self.bg_samples = np.zeros(
            (self.img_height,self.img_width,config.number_samples,config.attribute),dtype = np.uint16)
        # Dmin long term and short term of 2D array
        self.dist_min_lt_array = np.zeros((self.img_height,self.img_width))
        self.dist_min_st_array = np.zeros((self.img_height,self.img_width))
        # last mask of 2D array, either forground or background, if bg, then true
        self.last_mask = np.full((self.img_height,self.img_width),True)
        # last lbsp values
        self.last_lbsp_values = np.zeros((self.img_height,self.img_width),dtype=np.uint16)
        # last rgb values, scaled in range (0,255)
        self.last_rgb_values = np.zeros((self.img_height,self.img_width),dtype=np.uint16)
        # threshold Rx
        self.model_Rx = np.ones((self.img_height,self.img_width))
        # update rate Tx
        self.model_Tx = np.ones((self.img_height,self.img_width))
        # auxiliary parameter
        self.model_Vx = np.ones((self.img_height,self.img_width))
        # block and grid`s configuration of GPU
        if config.using_gpu:  
            self.BLOCKDIM = (16,8)
            blockspergrid_w = math.ceil(self.img_width/self.BLOCKDIM[0])
            blockspergrid_h = math.ceil(self.img_height/self.BLOCKDIM[1])
            self.GRIDDIM = (blockspergrid_w,blockspergrid_h)

        self.__init_bg_samples(init_image)

    def __compute_RGB_values(self,current_image)->np.array:

        rgb_values = np.zeros((self.img_height,self.img_width))
        if self.img_channel == 3: 
            rgb_values = np.sum(current_image,axis=2)//3
        elif self.img_channel == 1:
            rgb_values = current_image.copy()
        else:
            raise TypeError("what the hell is the image`s shape.")
        return rgb_values
    #@timer
    def __compute_LBSP_values(self,rgb_values_image)->np.array:
        lbsp_values =  np.zeros((self.img_height,self.img_width),dtype=np.uint16)
        if config.using_gpu:
            if cuda.is_available():
                lbsp.compute_lbsp_with_GPU(rgb_values_image,lbsp_values,self.BLOCKDIM,self.GRIDDIM)
            else:
                raise Exception("no available GPU device...")
        else:
            lbsp.compute_lbsp_without_GPU(rgb_values_image,lbsp_values)
        return lbsp_values

    def __init_bg_samples(self,init_image):
        rgb_values = self.__compute_RGB_values(init_image)
        init_lbsp = self.__compute_LBSP_values(rgb_values)

        for i in range(config.number_samples):
            self.bg_samples[:,:,i,0] = rgb_values.copy()
            self.bg_samples[:,:,i,1] = init_lbsp.copy()

    #@timer                
    def __update_Dmin_and_matching_BG(self,current_image,current_rgb_values,current_lbsp_values):
        
        dist_min_normalized = np.zeros((self.img_height,self.img_width))
        mask = np.full((self.img_height,self.img_width),False)

        if config.using_gpu:
            sample_rgb_values = np.ascontiguousarray(self.bg_samples)

            normalized_min_dist_and_match_gpu_kernel[self.GRIDDIM, self.BLOCKDIM](
                mask,
                dist_min_normalized,
                self.model_Rx,
                sample_rgb_values,
                current_rgb_values,
                current_lbsp_values)
        
        else:
            normalized_min_dist_and_match(
                mask,
                dist_min_normalized,
                self.model_Rx,
                self.bg_samples,
                current_rgb_values,
                current_lbsp_values)

        self.dist_min_st_array = self.dist_min_st_array*(1 - config.learning_rate_st) + \
                                    dist_min_normalized*config.learning_rate_st
        self.dist_min_st_array = np.where(self.dist_min_st_array>1.0,1.0,self.dist_min_st_array)

        self.dist_min_lt_array = self.dist_min_lt_array*(1 - config.learning_rate_lt) + \
                                    dist_min_normalized*config.learning_rate_lt
        self.dist_min_lt_array = np.where(self.dist_min_lt_array>1.0,1.0,self.dist_min_lt_array)
        
        #dmin = batch_max(self.dist_min_lt_array,self.dist_min_st_array)
        #save_path = os.path.join('./dmin_image/', path.split('/')[-1])
        #io.imsave(save_path,dist_min_normalized)
        return mask

    def __choose_updated_samples(self,mask:np.array,time_subsamples:np.array)->np.array:
        height,width = mask.shape

        if_updates = utils_func.pixel_updated_probability(time_subsamples.astype(np.int32))
        if_updates = np.where(mask==True,if_updates,False)
        which_samples = utils_func.random_choose_sample(np.full(mask.shape,config.number_samples))
        which_samples = np.where(if_updates==True,which_samples,-1)
        return which_samples

    @autojit
    def __update_samples_and_neighboor(self,
                                        bg_samples,
                                        which_samples:np.array,
                                        time_subsamples:np.array,
                                        current_image_rgb:np.array,
                                        current_image_lbsp:np.array):
        height,width = which_samples.shape
        for w in range(width):
            for h in range(height):
                which_sample = which_samples[h,w]
                if which_sample >= 0:
                    bg_samples[h,w,which_sample,0] = current_image_rgb[h,w]
                    bg_samples[h,w,which_sample,1] = current_image_lbsp[h,w]
                    
                    y,x = utils_func.random_choose_neighboor(h,w,width,height,int(time_subsamples[h,w]))
                    if w!=x or h!=y :
                        bg_samples[y,x,which_sample,0] = current_image_rgb[h,w]
                        bg_samples[y,x,which_sample,1] = current_image_lbsp[h,w]

    def __update_samples(self,
                        mask:np.array,
                        time_subsamples:np.array,
                        current_image_rgb:np.array,
                        current_image_lbsp:np.array):
        which_samples = self.__choose_updated_samples(mask,time_subsamples)
        self.__update_samples_and_neighboor(self.bg_samples,
                                            which_samples,
                                            time_subsamples,
                                            current_image_rgb,
                                            current_image_lbsp)

    def __compute_parameter_Vx(self,mask):
        blinks = mask ^ self.last_mask
        unstable_reg = np.where(
            utils_func.batch_min(self.dist_min_st_array,self.dist_min_lt_array)>
                config.unstable_reg_threshold,True,False)

        self.model_Vx += np.where(blinks==True,config.Vx_INCR,config.Vx_DECR)

        self.model_Vx = np.where(self.model_Vx<1,1.0,self.model_Vx) 
        self.model_Vx = np.where(self.model_Vx>100,100.0,self.model_Vx) 

    def __compute_parameter_Rx(self):

        dist_min = utils_func.batch_min(self.dist_min_st_array,self.dist_min_lt_array)
        mask = np.where(self.model_Rx<(1+dist_min*2)**2,True,False)
        self.model_Rx += np.where(mask==True,(0.01*(self.model_Vx-15)),-1/(self.model_Vx))
        #model_Rx += np.where(mask==True,model_Vx*0.02,-1/model_Vx)

        self.model_Rx = np.where(self.model_Rx<1.0,1.0,self.model_Rx)
        self.model_Rx = np.where(self.model_Rx>5.0,5.0,self.model_Rx)

    def __compute_parameter_Tx(self,mask):
        max_Dmin = utils_func.batch_max(self.dist_min_lt_array,self.dist_min_st_array)
        max_Dmin = np.where(max_Dmin<0.001,0.001,max_Dmin)

        multi = max_Dmin*self.model_Vx*0.01
        multi = np.where(multi<0.004,0.004,multi)
        dev = (self.model_Vx)/max_Dmin

        self.model_Tx += np.where(mask==False,0.5/multi,-1*dev)
        #print(np.max(0.1/multi),np.min(0.1/multi))                 
        #stable_reg = np.where(batch_min(dist_min_st_array,dist_min_lt_array)<UNSTABLE_REG_THRESHOLD,True,False)

        self.model_Tx = np.where(self.model_Tx<config.Tx_lower,config.Tx_lower,self.model_Tx)
        self.model_Tx = np.where(self.model_Tx>config.Tx_upper,config.Tx_upper,self.model_Tx)

    def __update_parameter(self,mask,rgb_values,lbsp_values):

        self.__compute_parameter_Vx(mask)
        self.__compute_parameter_Rx()
        self.__compute_parameter_Tx(mask)

        self.last_mask = mask.copy()
        self.last_rgb_values = rgb_values.copy()
        self.last_lbsp_values = lbsp_values.copy()


    def segmentation_image(self,mask)->np.array:
        img_shape = mask.shape
        #mask_image = is_matched_BG_samples_compile_optimal(current_image)
        seg_image = np.zeros((img_shape[0],img_shape[1]),dtype = np.uint8)
        forground = np.full((img_shape[0],img_shape[1]),255,dtype = np.uint8)

        seg_image = np.where(mask==False,forground,0)

        '''
        if len(img_shape) == 3: 
            for channel in range(3):
                seg_image[:,:,channel] = np.where(mask_image==False,current_image[:,:,channel],0)
        else:
            seg_image = np.where(mask_image==False,current_image,0)
        '''
        return seg_image

    #@timer
    def ones_iteration(self,current_image):

        rgb_values  = self.__compute_RGB_values(current_image)
        lbsp_values = self.__compute_LBSP_values(rgb_values)
        mask = self.__update_Dmin_and_matching_BG(current_image,rgb_values,lbsp_values)
        self.__update_samples(mask,self.model_Tx,rgb_values,lbsp_values)
        self.__update_parameter(mask,rgb_values,lbsp_values)
        seg_image = self.segmentation_image(mask)
        self.current_frame += 1
        return seg_image

        
    def debug_one_pixel_print(self,h=0,w=0):
        frame = {"RGB":self.bg_samples[h,w,:,0],
                "LBSP":self.bg_samples[h,w,:,1]}
        cell = DataFrame(frame,index=list(range(0,config.number_samples)))
        print(cell)



