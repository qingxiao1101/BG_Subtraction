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
from numba import vectorize, float64, int64,float32,int32,uint16,boolean
import random
import math

#random.seed(0)

file_path = './test_input/input/'

NUM_SAMPLE = 50
ATTRIBUTE = 2
RGB_THRESHOLD = 23
HAMMING_THRESHOLD = 4
MATCH_THRESHOLD = 2

USE_GPU = True
BLOCKDIM = (32,16)
GRIDDIM = (32,16)


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
	init_lbsp = compute_LBSP_values(rgb_values)

	for i in range(NUM_SAMPLE):
		bg_samples[:,:,i,0] = rgb_values.copy()
		bg_samples[:,:,i,1] = init_lbsp.copy()

#@timer
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
#@timer
def compute_LBSP_values(rgb_values_image)->np.array:
	lbsp_values =  np.zeros(rgb_values_image.shape,dtype=np.uint16)
	if USE_GPU:
		lbsp.lbsp_image_gpu(rgb_values_image,lbsp_values)
	else:
		lbsp.lbsp_image_normal(rgb_values_image,lbsp_values)
	return lbsp_values


#useless
#@timer
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

#@timer
@autojit
def is_matched_BG_samples_compile_optimal(current_image,
											current_rgb_values,
											current_lbsp_values)->np.array:
	
	mark_match = np.full(current_rgb_values.shape,False)
	for w in range(img_width):
		for h in range(img_height):
			mark_match[h,w] = match_one_sample(w,h,current_rgb_values[h,w],current_lbsp_values[h,w])

	return mark_match	


#-----------------------------------------------------------------------------------------

Dmin = np.zeros((img_height,img_width))
alpha = 0.4


def  normalization_min_distance(w,h,current_pixel):
	sample_rgb_values = bg_samples[h,w,:,0]
	sample_lbsp_values = bg_samples[h,w,:,1]
	curent_rgb_values = np.full(sample_rgb_values.shape,current_pixel)
	rgb_L1_dist = dist_L1_batch(sample_rgb_values,curent_rgb_values)
	return np.min(rgb_L1_dist)

@jit(nopython=True)
def  normalization_min_distance_compile_optimal(w,h,current_pixel):
	sample_rgb_values = bg_samples[h,w,:,0]
	sample_lbsp_values = bg_samples[h,w,:,1]
	min_dx = 255
	for sample in range(NUM_SAMPLE):
		current_dist = dist_L1_batch(sample_rgb_values[sample],current_pixel)
		if min_dx > current_dist:
			min_dx = current_dist
	return min_dx

#@timer
def update_Dmin(current_image):
	current_rgb_values = compute_RGB_values(current_image)
	rgb_L1_distances = np.zeros(current_rgb_values.shape)
	for w in range(img_width):
		for h in range(img_height):
			rgb_L1_distances[h,w] = normalization_min_distance_compile_optimal(w,h,current_rgb_values[h,w])

	dx = rgb_L1_distances/np.max(rgb_L1_distances)
	global Dmin
	Dmin = Dmin*(1 - alpha) + dx*alpha


def  device_normalization_min_distance(samples,current_rgb,current_lbsp):

	min_dx = 1
	for idx in range(NUM_SAMPLE):
		rgb_dist = abs(samples[idx,0] - current_rgb)
		
		lbsp_dist = 0
		exr = samples[idx,1]+current_lbsp
		for i in range(16):
			if (exr>>i)&1 == 1:
				lbsp_dist += 1
		
		#dist = (rgb_dist/256 + lbsp_dist/64)/2
		dist = rgb_dist/256
		if min_dx > dist:
			min_dx = dist
	return min_dx
	
normalization_min_distance_gpu = cuda.jit(device=True)(device_normalization_min_distance)
@cuda.jit
def update_Dmin_gpu_kernel(distances,sample_values,current_rgb_values,current_lbsp_values):
	height = current_rgb_values.shape[0]
	width = current_rgb_values.shape[1]
	
	startX, startY = cuda.grid(2)
	gridX = cuda.gridDim.x * cuda.blockDim.x;
	gridY = cuda.gridDim.y * cuda.blockDim.y;
	for x in range(startX, width, gridX):
		for y in range(startY, height, gridY): 
			distances[y,x] = normalization_min_distance_gpu(sample_values[y,x],np.int64(current_rgb_values[y,x]),current_lbsp_values[y,x])
	

#@timer
def update_Dmin_gpu(current_image,current_rgb_values,current_lbsp_values):
	dx = np.zeros(current_rgb_values.shape)
	sample_rgb_values = np.ascontiguousarray(bg_samples[:,:,:,:])
	update_Dmin_gpu_kernel[GRIDDIM, BLOCKDIM](dx,sample_rgb_values,current_rgb_values,current_lbsp_values)
	global Dmin
	Dmin = Dmin*(1 - alpha) + dx*alpha
#-----------------------------------------------------------------------------------------
	

def segmentation_image(mask)->np.array:
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
#-----------------------------------------------------------------------------------------
'''
	- update samples
'''		
@vectorize([boolean(int32),
		boolean(int64)])
def pixel_updated_probability(t):
	if (random.randint(0,t-1)<1):
		return True
	else:
		return False

@vectorize([int64(int32),
		int64(int64)])		
def random_choose_sample(t):
	return random.randint(0,t-1)

@jit
def random_choose_neighboor(h,w,img_width,img_height,t)->tuple:
	if random.randint(0,t-1) < 1 :
		idx = random.randint(0,7)
		if idx==0 and h>0 and w>0:
			return h-1,w-1
		elif idx ==1 and h>0:
			return h-1,w 
		elif idx ==2 and h>0 and w<img_width-1:
			return h-1,w+1
		elif idx ==3 and w>0:
			return h,w-1
		elif idx ==4 and w<img_width-1:
			return h,w+1
		elif idx ==5 and h<img_height-1 and w>0:
			return h+1,w-1
		elif idx ==6 and h<img_height-1:
			return h+1,w 
		elif idx ==7 and h<img_height-1 and w<img_width-1:
			return h+1,w+1
		else:
			return h,w
	else:
		return h,w

def choose_updated_samples(mask:np.array,time_subsamples:np.array)->np.array:
	height,width = mask.shape
	if_updates = pixel_updated_probability(time_subsamples)
	if_updates = np.where(mask==True,if_updates,False)
	which_samples = random_choose_sample(np.full(mask.shape,NUM_SAMPLE))
	which_samples = np.where(if_updates==True,which_samples,-1)

	return which_samples

@autojit
def update_samples_and_neighboor(bg_samples,
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
				y,x = random_choose_neighboor(h,w,width,height,time_subsamples[h,w])
				if w!=x or h!=y :
					pass
					bg_samples[y,x,which_sample,0] = current_image_rgb[h,w]
					bg_samples[y,x,which_sample,1] = current_image_lbsp[h,w]

#@timer
def update_samples(mask:np.array,
					time_subsamples:np.array,
					current_image_rgb:np.array,
					current_image_lbsp:np.array):
	which_samples = choose_updated_samples(mask,time_subsamples)
	update_samples_and_neighboor(bg_samples,which_samples,time_subsamples,current_image_rgb,current_image_lbsp)

#-----------------------------------------------------------------------------------------


def debug_one_pixel_print(h=0,w=0):
	frame = {"RGB":bg_samples[h,w,:,0],
			"LBSP":bg_samples[h,w,:,1]}
	cell = DataFrame(frame,index=list(range(0,NUM_SAMPLE)))
	print(cell)

@timer
def ones_iteration(current_image,T_subsamples):
	rgb_values  = compute_RGB_values(current_image)
	lbsp_values = compute_LBSP_values(rgb_values)
	mask        = is_matched_BG_samples_compile_optimal(current_image,rgb_values,lbsp_values)
	update_samples(mask,T_subsamples,rgb_values,lbsp_values)
	update_Dmin_gpu(current_image,rgb_values,lbsp_values)


if __name__ == '__main__':
	init_bg_samples()
	subsamples = np.full((img_height,img_width),5)
	#pre_rgb_values = np.full((img_height,img_width),0)
	img = read_one_image(img_paths[0])
		#ones_iteration(image,subsamples)
	rgb_value  = compute_RGB_values(img)
	pre_lbsp_values = compute_LBSP_values(rgb_value)
	#pre_lbsp_values = np.full((img_height,img_width),0)
	for path in img_paths:
		print(path,"-------------------------------------------#")
		image = read_one_image(path)
		ones_iteration(image,subsamples)

		dmin_path = os.path.join('./dmin_image/', path.split('/')[-1])
		io.imsave(dmin_path,lbsp_dist)
		
	