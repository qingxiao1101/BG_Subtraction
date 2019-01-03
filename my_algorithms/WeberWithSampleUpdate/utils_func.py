#!/usr/bin/env python3

import numpy as np
import random
from numba import vectorize,float64,int64,float32,int32,uint16,boolean
from numba import autojit, jit, cuda
from time import time
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

@jit
def get_neighboor_coordinate(h,w,img_width,img_height,idx):
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

@autojit
def device_choose_neighboor(h,w,img_width,img_height,idx):
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



@autojit
def device_compute_one_pixel(samples,
								weights,
								threshold_Rx,
								if_update_sample,
								current_rgb,
								stochastic,
								which_sample):
	match_weberrate_times = 0
	min_weber_rate = 1.0
	min_weight = config.weight_upper
	second_min_weight = config.weight_upper
	min_weight_idx = 0
	second_min_weight_idx = 0
	max_weight = 0
	sum_weight = 0
	max_weber_rate = 0.0
	max_weber_idx = 0
	sum_weber = 0
	min_rgb = 255

	current_value = float(current_rgb[0]+current_rgb[1]+current_rgb[2])/3

	#computing saturation of one
	max_rgb = max(max(current_rgb[0],current_rgb[1]),current_rgb[2])/255.0
	min_rgb = min(min(current_rgb[0],current_rgb[1]),current_rgb[2])/255.0
	C = max_rgb - min_rgb
	V = 0.5*(max_rgb + min_rgb)
	L = 0
	if V<=config.a:
		L = V*(1/config.a)
	else:
		L = (1-V)*(1/(1-config.a))
	S = min(C/L,1.0)

	for idx in range(config.number_samples):
		# type uint8 must be converted to float
		sum_weight += weights[idx]
		sample_value = float(samples[idx,0] + samples[idx,1] + samples[idx,2])/3
		
		rgb_dist = abs(sample_value - current_value)
		#weber_rate = min(1.0, 9*rgb_dist/current_value**1.5) #theshold<0.26
		#weber_rate = min(1.0, rgb_dist/sample_value)
		#weber_rate = min(1.0, 25*3*rgb_dist/min(current_value,sample_value)**2)  #theshold>0.15
		weber_rate = min(1.0, 9*rgb_dist/min(current_value,sample_value)**1.5) #theshold<0.13

		weber_rate = weber_rate*(S**1.2)

		sum_weber += weber_rate
		#weights[idx] += 1
			
		if rgb_dist <= 23:#*threshold_Rx:#better
		#if weber_rate < 0.25:
			match_weberrate_times += 1
			weights[idx] += 2
			'''
			if stochastic <= config.weight_incr:
				if samples[idx,3] < config.weight_upper-1:
					samples[idx,3] += 1
			'''
		else:
			#pass
			weights[idx] -= 1
			#weights[idx] -= 5
			#if stochastic <= config.weight_decr:
			#	if samples[idx,3] > config.weight_lower+3:
			#		samples[idx,3] -= 3
		if min_rgb >= rgb_dist:
			min_rgb = rgb_dist
		if max_weber_rate <= weber_rate:
			max_weber_rate = weber_rate
			max_weber_idx = idx

		if min_weber_rate >= weber_rate:
			min_weber_rate = weber_rate
		if max_weight <= weights[idx]:
			max_weight = weights[idx]
		if min_weight >= weights[idx]:
			second_min_weight = min_weight
			second_min_weight_idx = min_weight_idx
			min_weight = weights[idx]
			min_weight_idx = idx
		elif second_min_weight >= weights[idx]:
			second_min_weight = weights[idx]
			second_min_weight_idx = idx
	if match_weberrate_times >= config.match_threshold:
		#background
		
		if if_update_sample==True:
			if config.if_stochastic_update==True:
				samples[which_sample,0] = current_rgb[0]
				samples[which_sample,1] = current_rgb[1]
				samples[which_sample,2] = current_rgb[2]
			else:
				samples[min_weight_idx,0] = current_rgb[0]
				samples[min_weight_idx,1] = current_rgb[1]
				samples[min_weight_idx,2] = current_rgb[2]
				weights[min_weight_idx] = second_min_weight+1#sum_weight//config.number_samples
				#+int((1/match_weberrate_times)*config.number_samples*10)
		#return True, min(1.0,sum_weber/config.number_samples), second_min_weight, second_min_weight_idx		
		return True, min_weber_rate, min_weight, min_weight_idx
				
	else:
		#forground
		if_update_sample = False
		#return False, min(1.0,sum_weber/config.number_samples), second_min_weight, second_min_weight_idx
		return False, min_weber_rate, min_weight, min_weight_idx

@autojit				
def device_update_neighboor(samples,weights, current_rgb, sum_weight, min_weight_idx):
	samples[min_weight_idx,0] = current_rgb[0]
	samples[min_weight_idx,1] = current_rgb[1]
	samples[min_weight_idx,2] = current_rgb[2]
	#if max_weight < config.weight_upper - 1:
	weights[min_weight_idx] = sum_weight+1# + 1






'''
	calculating minimal normalized distance and forground or background judgement for one pixel
'''
@autojit
def match_and_normalizd_min_dist_back_up(samples,threshold_Rx,current_rgb,current_lbsp):
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
def compute_one_pixel_gpu_kernel(mask,
									weber_rate,
									max_weight,
									min_weight_idx,
									model_Rx,
									if_update_sample,
									if_update_neighboor,
									bg_samples,
									bg_weights,
									current_rgb_values,
									stochastic,
									neighboor_idx,
									which_samples):
											
	height = current_rgb_values.shape[0]
	width = current_rgb_values.shape[1]

	abs_X,abs_Y = cuda.grid(2)  #absolute position 
	n_y = 0
	n_x = 0
	if abs_X < width and abs_Y < height:

		mask[abs_Y,abs_X], weber_rate[abs_Y,abs_X], \
		max_weight[abs_Y,abs_X],min_weight_idx[abs_Y,abs_X] = \
			device_compute_one_pixel(
				bg_samples[abs_Y,abs_X],
				bg_weights[abs_Y,abs_X],
				model_Rx[abs_Y,abs_X],
				if_update_sample[abs_Y,abs_X],
				current_rgb_values[abs_Y,abs_X],
				stochastic[abs_Y,abs_X],
				which_samples[abs_Y,abs_X])
			
		if not config.if_stochastic_update and config.if_update_neighboor == True:
			if if_update_sample[abs_Y,abs_X] and if_update_neighboor[abs_Y,abs_X] ==True: #and if_update_sample[abs_Y,abs_X] 
				n_y,n_x = device_choose_neighboor(abs_Y,abs_X,width,height,neighboor_idx[abs_Y,abs_X])
				device_update_neighboor(
					bg_samples[n_y,n_x],
					bg_weights[abs_Y,abs_X],
					current_rgb_values[abs_Y,abs_X],
					max_weight[n_y,n_x],
					min_weight_idx[n_y,n_x])
		elif config.if_stochastic_update and config.if_update_neighboor == True:
			if if_update_sample[abs_Y,abs_X] and if_update_neighboor[abs_Y,abs_X] ==True: #and if_update_sample[abs_Y,abs_X] 
				n_y,n_x = device_choose_neighboor(abs_Y,abs_X,width,height,neighboor_idx[abs_Y,abs_X])
				device_update_neighboor(
					bg_samples[n_y,n_x],
					bg_weights[n_y,n_x],
					current_rgb_values[abs_Y,abs_X],
					max_weight[n_y,n_x],
					which_samples[n_y,n_x])
		
		

		


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



