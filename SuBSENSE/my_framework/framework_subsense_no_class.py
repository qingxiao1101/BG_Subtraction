#!/usr/bin/env python3
import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from skimage import io, transform
from pylab import imshow, show
from pandas import Series,DataFrame
from time import time
from dist_utils import dist_L1_batch, dist_hamming_batch
import lbsp
from numba import autojit, jit, cuda

file_path = './test_input/'

NUM_SAMPLE = 50
ATTRIBUTE = 2
RGB_THRESHOLD = 23
HAMMING_THRESHOLD = 4
MATCH_THRESHOLD = 2

USE_GPU = True

def timer(func):
    def deco(*args, **kwargs):  
        start = time()
        res = func(*args, **kwargs)
        stop = time()
        print('function (%s) cost %f seconds' %(func.__name__,stop-start))
        return res 
    return deco

def read_image_paths(filePath = file_path )->[str]: 
	img_names = [x for x in os.listdir(filePath) if x.split('.')[-1] in 'jpg|png']
	img_names = sorted(img_names, key = lambda x: int(x.split('.')[-2][2:]))
	img_paths = [os.path.join(filePath, x) for x in img_names]
	return img_paths

def read_one_image(imgPath) -> np.array: #return uint8
	img = cv2.imread(imgPath)
	return img

'''
	- some global variables initialization
'''
img_paths = read_image_paths()
img_height,img_width,img_channel = read_one_image(img_paths[0]).shape
bg_samples = np.zeros((img_height,img_width,NUM_SAMPLE,ATTRIBUTE),dtype = int)

def init_bg_samples():
	first_image = read_one_image(img_paths[0])
	rgb_values = compute_RGB_values(first_image)
	init_lbsp = compute_LBSP_values(first_image)

	for i in range(NUM_SAMPLE):
		bg_samples[:,:,i,0] = rgb_values.copy()
		bg_samples[:,:,i,1] = init_lbsp.copy()


def compute_RGB_values(current_image)->np.array:
	img_shape = current_image.shape
	rgb_values = np.zeros((current_image.shape[0],current_image.shape[1]))
	if len(img_shape) == 3: 
		rgb_values = np.sum(current_image,axis=2)//3
	elif len(img_shape) == 2:
		rgb_values = current_image.copy()
	else:
		raise TypeError("shape of input image error.")
	return rgb_values

def compute_LBSP_values(current_image)->np.array:
	lbsp_values =  np.zeros((current_image.shape[0],current_image.shape[1]),dtype=np.uint16)
	if USE_GPU:
		lbsp.lbsp_image_gpu(compute_RGB_values(current_image),lbsp_values)
	else:
		lbsp.lbsp_image_normal(compute_RGB_values(current_image),lbsp_values)
	return lbsp_values

@timer
@jit
def is_matched_BG_samples(current_image)->np.array:

	current_rgb_values = compute_RGB_values(current_image)
	current_lbsp_values =  compute_LBSP_values(current_image)
	mark_rgb_match = np.full(current_rgb_values.shape,0)
	mark_lbsp_match = np.full(current_lbsp_values.shape,0)
	for idx in range(NUM_SAMPLE):
		sample_rgb_values = bg_samples[:,:,idx,0] 
		sample_lbsp_values = bg_samples[:,:,idx,1] 
		mark_rgb_match = mark_rgb_match + np.where(dist_L1_batch(sample_rgb_values,current_rgb_values) <= RGB_THRESHOLD,1,0)
		mark_lbsp_match = mark_lbsp_match + np.where(dist_hamming_batch(sample_lbsp_values,current_lbsp_values) <= HAMMING_THRESHOLD,1,0)
	return np.logical_and(np.where(mark_rgb_match>=MATCH_THRESHOLD,True,False),np.where(mark_lbsp_match>=MATCH_THRESHOLD,True,False))

@jit(nopython=True)
def match_one_sample(width,height,current_rgb_value,current_lbsp_value)->bool:
	match_rgb_times = 0
	match_lbsp_times = 0
	for idx in range(NUM_SAMPLE):
		sample_rgb_value = bg_samples[height,width,idx,0]
		if dist_L1_batch(sample_rgb_value,current_rgb_value)<= RGB_THRESHOLD:
			match_rgb_times += 1
			if match_rgb_times >= MATCH_THRESHOLD :
				sample_lbsp_value = bg_samples[height,width,idx,1] 
				if dist_hamming_batch(sample_lbsp_value,current_lbsp_value) <= HAMMING_THRESHOLD:
					match_lbsp_times += 1 
					if 	match_lbsp_times >= MATCH_THRESHOLD :
						return True

	return False

@timer
@autojit
def is_matched_BG_samples_compile_optimal(current_image)->np.array:
	current_rgb_values = compute_RGB_values(current_image)
	current_lbsp_values =  compute_LBSP_values(current_image)
	mark_match = np.full(current_rgb_values.shape,False)
	
	for w in range(img_width):
		for h in range(img_height):
			mark_match[h,w] = match_one_sample(w,h,current_rgb_values[h,w],current_lbsp_values[h,w])

	return mark_match		

def segmentation_image(current_image)->np.array:
	img_shape = current_image.shape
	mask_image = is_matched_BG_samples_compile_optimal(current_image)
	seg_image = np.zeros((img_shape[0],img_shape[1]),dtype = np.uint8)
	forground = np.full((img_shape[0],img_shape[1]),255,dtype = np.uint8)

	seg_image = np.where(mask_image==False,forground,0)

	'''
	if len(img_shape) == 3: 
		for channel in range(3):
			seg_image[:,:,channel] = np.where(mask_image==False,current_image[:,:,channel],0)
	else:
		seg_image = np.where(mask_image==False,current_image,0)
	'''
	return seg_image


def debug_one_pixel_print(self,h=0,w=0):
	frame = {"RGB":self.bg_samples[h,w,:,0],
			"LBSP":self.bg_samples[h,w,:,1]}
	cell = DataFrame(frame,index=list(range(1,NUM_SAMPLE+1)))
	print(cell)


if __name__ == '__main__':

    image = read_one_image(img_paths[1])
    init_bg_samples()
    #mark = is_matched_BG_samples_compile_optimal(image)
    is_matched_BG_samples_compile_optimal(image)
    test = segmentation_image(image)


    io.imshow(test)
    io.show()
    #cv2.imshow("test",test)
    #cv2.waitKey(0)


