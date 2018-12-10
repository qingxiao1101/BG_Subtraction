#!/usr/bin/env python3

import numpy as np 
import random
from numba import vectorize, float64, int64,float32,int32,uint16, boolean
from numba import autojit, jit, cuda
from time import time
import configuration as config

def timer(func):
    def deco(*args, **kwargs):  
        start = time()
        res = func(*args, **kwargs)
        stop = time()
        print('function (%s) cost %f seconds' %(func.__name__,stop-start))
        return res 
    return deco

@vectorize([boolean(int32),
		boolean(int64)])
def pixel_updated_probability(t):
	if (random.randint(0,int(t)-1)<1):
		return True
	else:
		return False

@vectorize([int32(int32),
		int64(int64)])		
def random_choose_sample(t):
	return random.randint(0,t-1)

'''
	input pixel`s coordinate and return randomly neighboor`s coordinate
'''
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



@vectorize([int32(int32, int32),
		int64(int64, int64),
		float32(float32, float32),
		float64(float64, float64)])
def dist_L1_batch(para1,para2):
	return abs(para1-para2)

@vectorize([uint16(uint16, uint16),
		int32(int32, int32),
		int64(int64, int64)]) #,target='cuda'
def dist_hamming_batch(para1,para2):
	dist = 0
	exr = para1^para2
	for idx in range(16):
		if (exr>>idx)&1 == 1:
			dist += 1
	return dist


@vectorize([int32(int32, int32),
		uint16(uint16, uint16),
		float32(float32, float32),
		float64(float64, float64)])
def batch_min(a,b):
	return min(a,b)
@vectorize([int32(int32, int32),
		uint16(uint16, uint16),
		float32(float32, float32),
		float64(float64, float64)])
def batch_max(a,b):
	return max(a,b)

'''
	calculating minimal normalized distance and forground or background judgement for one pixel
'''
@autojit
def match_and_normalizd_min_dist(samples,threshold_Rx,current_rgb,current_lbsp):
	match_rgb_times = 0
	match_lbsp_times = 0
	min_rgb_dist = 255
	min_sum_dist = 255
	min_lbsp_dist = 16
	sum_dist = 0
	for idx in range(config.number_samples): 
		# type uint16 must be converted to float
		rgb_dist = abs(float(samples[idx,0]) - float(current_rgb))
		lbsp_dist = 0
		exr = samples[idx,1] ^ current_lbsp
		for i in range(16):
			if (exr>>i)&1 == 1:
				lbsp_dist += 1

		if rgb_dist <= config.rgb_init_threshold*threshold_Rx:
			match_rgb_times += 1
			if lbsp_dist <= config.lbsp_init_threshold+2**threshold_Rx: 
				match_lbsp_times += 1

		sum_dist = min((lbsp_dist/4)*(255/16)+rgb_dist, 255)
		if min_rgb_dist > rgb_dist:
			min_rgb_dist = rgb_dist
		if min_lbsp_dist > lbsp_dist:
			min_lbsp_dist = lbsp_dist
		if min_sum_dist > sum_dist:
			min_sum_dist = sum_dist
		#min_sum_norm_dist = min(1.0, MIN_NORM_DIST_INTENSITY*((sum_dist/256) + (min_lbsp_dist/16)))
	if match_lbsp_times >= config.match_threshold:
			#background
			#return True, min(1.0, (min_rgb_dist/256 + min_lbsp_dist/16)/2)
		return True, min(1.0, 0.5*((min_sum_dist/256) + (min_lbsp_dist/16)))
	else:
			#forground
			#+ (2-match_lbsp_times)/4(MATCH_THRESHOLD-match_lbsp_times)/MATCH_THRESHOLD
		return False, min(1.0, 0.5*((min_sum_dist/256) + (min_lbsp_dist/16))+(2-match_lbsp_times)/2) 


@autojit
def normalized_min_dist_and_match(mask,
								distances,
								model_Rx,
								sample_values,
								current_rgb_values,
								current_lbsp_values):
	height = current_rgb_values.shape[0]
	width = current_rgb_values.shape[1]

	for w in range(width):
		for h in range(height):
			mask[h,w],distances[h,w] = match_and_normalizd_min_dist(
				sample_values[h,w],
				model_Rx[h,w],
				np.uint16(current_rgb_values[h,w]),
				current_lbsp_values[h,w])

@cuda.jit
def normalized_min_dist_and_match_gpu_kernel(mask,
											distances,
											model_Rx,
											sample_values,
											current_rgb_values,
											current_lbsp_values):

	height = current_rgb_values.shape[0]
	width = current_rgb_values.shape[1]

	abs_X,abs_Y = cuda.grid(2)  #absolute position 
	if abs_X < width and abs_Y < height:
		mask[abs_Y,abs_X],distances[abs_Y,abs_X] = match_and_normalizd_min_dist(
			sample_values[abs_Y,abs_X],
			model_Rx[abs_Y,abs_X],
			np.uint16(current_rgb_values[abs_Y,abs_X]),
			current_lbsp_values[abs_Y,abs_X])


"""
class Distance():
	def __init__(self):
		raise TypeError('class Distance not instantiable...')
		
	#this function useless
	def dist_L1(para1,para2):
		if isinstance(para1,np.ndarray) and isinstance(para2,np.ndarray):
			return np.abs(para1-para2)
		elif isinstance(para1,float) and isinstance(para2,float):
			return abs(para1-para2)
		elif isinstance(para1,int) and isinstance(para2,int):
			return abs(para1-para2)
		else:
			raise("parametric type not acceptable...")



	@staticmethod
	def dist_L1_batch(para1,para2):
		para1 = np.array(para1)
		para2 = np.array(para2)
		return np.abs(para1 - para2)


	@vectorize([int32(int32, int32),
            int64(int64, int64),
            float32(float32, float32),
            float64(float64, float64)])
	def dist_L1_batch_optimal(para1,para2):
		return abs(para1-para2)
	
	#this function useless
	#@vectorize([int32(int32, int32),
     #       int64(int64, int64)],target='cuda')	
	def dist_Hamming(para1,para2):
		para1 = np.array(para1,dtype=np.uint16)
		para2 = np.array(para2,dtype=np.uint16)
		return bin(para1^para2).count('1') #bitwise exclusive or

	@staticmethod
	def dist_hamming_batch(para1,para2):

		para1 = np.array(para1,dtype=np.uint16)
		para2 = np.array(para2,dtype=np.uint16)

		#result of para1^para2 is a vector of exclusive_or
		bins = np.vectorize(lambda x: bin(x))(para1^para2)
		dist = np.vectorize(lambda x: x.count('1'))(bins)

		return dist


	@vectorize([uint16(uint16, uint16),
			int32(int32, int32),
            int64(int64, int64)]) #,target='cuda'
	def dist_hamming_batch_optimal(para1,para2):
		dist = 0
		exr = para1^para2
		for idx in range(16):
			if (exr>>idx)&1 == 1:
				dist += 1
		return dist



"""
"""
@timer
def test():
	np.random.seed(0)
	x = np.random.randint(1,10,(480,720))
	y = np.random.randint(100,110,(480,720))
	t = np.zeros(x.shape)
	for i in range(50):
		t = dist_hamming_batch_optimal(x,y)
	return t

if __name__ == '__main__':

	print(test())
"""



