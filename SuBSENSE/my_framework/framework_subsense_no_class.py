#!/usr/bin/env python3
import os
import cv2
import random
import math
import numpy as np 
import matplotlib.pyplot as plt 
from skimage import io, transform,exposure
from pylab import imshow, show
from pandas import Series,DataFrame
from time import time
from numba import autojit, jit, cuda
from numba import vectorize, float64, int64,float32,int32,uint16,int16, boolean
from dist_utils import dist_L1_batch, dist_hamming_batch
import lbsp
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
#random.seed(0)

file_path = './test_input/input_fall/'
#file_path = './test_input/input_office/'
#file_path = './test_input/input_highway/'


NUM_SAMPLE = 50
ATTRIBUTE = 2
RGB_THRESHOLD = 30
HAMMING_THRESHOLD = 4
MATCH_THRESHOLD = 2
CURRENT_FRAME = 0



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
bg_samples = np.zeros((img_height,img_width,NUM_SAMPLE,ATTRIBUTE),dtype = np.uint16)


USE_GPU = True

BLOCKDIM = (16,8)
blockspergrid_w = math.ceil(img_width/BLOCKDIM[0])
blockspergrid_h = math.ceil(img_height/BLOCKDIM[1])
GRIDDIM = (blockspergrid_w,blockspergrid_h)

# Dmin long term and short term of 2D array
dist_min_lt_array = np.zeros((img_height,img_width))
dist_min_st_array = np.zeros((img_height,img_width))

dist_min_normalized = np.zeros((img_height,img_width))

# last mask of 2D array, either forground or background, if bg, then true
last_mask = np.full((img_height,img_width),True)
# last lbsp values
last_lbsp_values = np.zeros((img_height,img_width),dtype=np.uint16)
# last rgb values, scaled in range (0,255)
last_rgb_values = np.zeros((img_height,img_width),dtype=np.uint16)
# learning rate alpha with long term and short term
LEARNING_RATE_ST = 0.04
LEARNING_RATE_LT = 0.01



def timer(func):
    def deco(*args, **kwargs):  
        start = time()
        res = func(*args, **kwargs)
        stop = time()
        print('function (%s) cost %f seconds' %(func.__name__,stop-start))
        return res 
    return deco


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

#useless
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
def update_D_MIN(current_image):
	current_rgb_values = compute_RGB_values(current_image)
	rgb_L1_distances = np.zeros(current_rgb_values.shape)
	for w in range(img_width):
		for h in range(img_height):
			rgb_L1_distances[h,w] = normalization_min_distance_compile_optimal(w,h,current_rgb_values[h,w])

	dx = rgb_L1_distances/np.max(rgb_L1_distances)
	global dist_min_st_array
	dist_min_st_array = dist_min_st_array*(1 - LEARNING_RATE_ST) + dx*LEARNING_RATE_ST

'''
	- return values: is matched and minimum normalized distance
	- in device GPU cannot use dist_utils functions
'''

#intensity adjustment for min normalized distance dx
MIN_NORM_DIST_INTENSITY = 2
#intensity adjustment for Dmin
DMIN_INTENSITY = 1.0

@jit(nopython=True)
def  device_normalization_min_distance(samples,current_rgb,current_lbsp):
	match_rgb_times = 0
	match_lbsp_times = 0
	min_rgb_dist = 255
	min_sum_dist = 255
	min_lbsp_dist = 16
	sum_dist = 0
	for idx in range(NUM_SAMPLE):
		# type uint16 must be converted to float
		rgb_dist = abs(float(samples[idx,0]) - float(current_rgb))
		lbsp_dist = 0
		exr = samples[idx,1] ^ current_lbsp
		for i in range(16):
			if (exr>>i)&1 == 1:
				lbsp_dist += 1

		if rgb_dist <= RGB_THRESHOLD:
			match_rgb_times += 1
			if lbsp_dist <= HAMMING_THRESHOLD:
				match_lbsp_times += 1

		sum_dist = min((lbsp_dist/4)*(255/16)+rgb_dist, 255)
		if min_rgb_dist > rgb_dist:
			min_rgb_dist = rgb_dist
		if min_lbsp_dist > lbsp_dist:
			min_lbsp_dist = lbsp_dist
		if min_sum_dist > sum_dist:
			min_sum_dist = sum_dist
	#min_sum_norm_dist = min(1.0, MIN_NORM_DIST_INTENSITY*((sum_dist/256) + (min_lbsp_dist/16)))
	if match_lbsp_times >= MATCH_THRESHOLD:
		#background
		#return True, min(1.0, (min_rgb_dist/256 + min_lbsp_dist/16)/2)
		return True, min(1.0, MIN_NORM_DIST_INTENSITY*((min_sum_dist/256) + (min_lbsp_dist/16)))
	else:
		#forground
		#return False, min(1.0, MIN_NORM_DIST_INTENSITY*((min_rgb_dist/256) + (min_lbsp_dist/16))) #+ (2-match_lbsp_times)/4(MATCH_THRESHOLD-match_lbsp_times)/MATCH_THRESHOLD
		return False, min(1.0, MIN_NORM_DIST_INTENSITY*((min_sum_dist/256) + (min_lbsp_dist/16))+(2-match_lbsp_times)/2) #+ (2-match_lbsp_times)/4(MATCH_THRESHOLD-match_lbsp_times)/MATCH_THRESHOLD


normalization_min_distance_gpu = cuda.jit(device=True)(device_normalization_min_distance)

@cuda.jit
def gpu_kernel_for_D_MIN_and_match(mask,distances,sample_values,current_rgb_values,current_lbsp_values):
	height = current_rgb_values.shape[0]
	width = current_rgb_values.shape[1]

	abs_X,abs_Y = cuda.grid(2)

	if abs_X < width and abs_Y < height:
			mask[abs_Y,abs_X],distances[abs_Y,abs_X] = normalization_min_distance_gpu(sample_values[abs_Y,abs_X],
																		np.uint16(current_rgb_values[abs_Y,abs_X]),
																		current_lbsp_values[abs_Y,abs_X])
	

@vectorize([int32(int32, int32),
		int16(int16, int16),
		float32(float32, float32),
		float64(float64, float64)])
def batch_min(a,b):
	return min(a,b)
@vectorize([int32(int32, int32),
		int16(int16, int16),
		float32(float32, float32),
		float64(float64, float64)])
def batch_max(a,b):
	return max(a,b)


def update_D_MIN_and_matching_BG_by_GPU(current_image,current_rgb_values,current_lbsp_values):
	global dist_min_normalized
	#normalized_min_dist = np.zeros(current_rgb_values.shape)
	mask = np.full(current_rgb_values.shape,False)
	sample_rgb_values = np.ascontiguousarray(bg_samples)
	gpu_kernel_for_D_MIN_and_match[GRIDDIM, BLOCKDIM](mask,dist_min_normalized,sample_rgb_values,current_rgb_values,current_lbsp_values)

	global dist_min_st_array
	dist_min_st_array = dist_min_st_array*(1 - LEARNING_RATE_ST) + dist_min_normalized*LEARNING_RATE_ST*DMIN_INTENSITY
	dist_min_st_array = np.where(dist_min_st_array>1.0,1.0,dist_min_st_array)

	global dist_min_lt_array
	dist_min_lt_array = dist_min_lt_array*(1 - LEARNING_RATE_LT) + dist_min_normalized*LEARNING_RATE_LT*DMIN_INTENSITY
	dist_min_lt_array = np.where(dist_min_lt_array>1.0,1.0,dist_min_lt_array)
	
	dmin = batch_max(dist_min_lt_array,dist_min_st_array)
	#save_path = os.path.join('./dmin_image/', path.split('/')[-1])
	#io.imsave(save_path,dist_min_normalized)
	#view_dx.append(dx[360,650])
	#print("D_MIN max:",np.max(D_MIN),"min:",np.min(D_MIN))
	return mask
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

@vectorize([int32(int32),
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

model_Vx = np.ones((img_height,img_width))
V_INCR = 1.0
V_DECR = -0.1
UNSTABLE_REG_THRESHOLD = 0.1
ACCUMULATE_NUM = 100
def compute_parameter_Vx(mask):

	global model_Vx, last_mask

	blinks = mask ^ last_mask
	unstable_reg = np.where(batch_min(dist_min_st_array,dist_min_lt_array)>UNSTABLE_REG_THRESHOLD,True,False)
	stable_reg = np.logical_not(unstable_reg)
	unblinks = np.logical_not(blinks)
	model_Vx += np.where(blinks==True,V_INCR,V_DECR)
	high_dist_min_normalized_reg = np.where(dist_min_normalized**2>0.5,True,False)
	model_Vx += np.where(high_dist_min_normalized_reg==True,V_DECR/2,0.0)
	#model_Vx += np.where(np.logical_and(blinks,unstable_reg)==True,V_INCR,V_DECR) 

	'''
	model_Vx += np.where(np.logical_and(blinks,unstable_reg)==True,V_INCR,0.0) 
	#model_Vx += np.where(np.logical_or(blinks,unstable_reg)==False,V_DECR,0.0) 
	model_Vx += np.where(np.logical_and(blinks,stable_reg)==True,V_DECR/4,0.0) 
	model_Vx += np.where(np.logical_and(unblinks,unstable_reg)==True,V_DECR/2,0.0) 
	model_Vx += np.where(np.logical_and(unblinks,stable_reg)==True,V_DECR,0.0) 
	'''

	model_Vx = np.where(model_Vx<1,1.0,model_Vx) 
	model_Vx = np.where(model_Vx>100,100.0,model_Vx) 
	#temp = np.where(blinks==True,0.99,0.0)
	#save_path = os.path.join('./dmin_image/', path.split('/')[-1])
	#max_value = np.max(model_Vx)
	#io.imsave(save_path,dist_min_normalized)


model_Rx = np.ones((img_height,img_width))


def compute_parameter_Rx():
	global model_Rx
	dist_min = batch_min(dist_min_st_array,dist_min_lt_array)
	#mask = np.where(dist_min_st_array>0.1,True,False)
	mask = np.where(model_Rx<(1+dist_min*2)**2,True,False)
	model_Rx += np.where(mask==True,(0.01*(model_Vx-15)),-1/(model_Vx))
	#model_Rx += np.where(mask==True,model_Vx*0.02,-1/model_Vx)

	model_Rx = np.where(model_Rx<1.0,1.0,model_Rx)
	model_Rx = np.where(model_Rx>6.0,6.0,model_Rx)
	#save_path = os.path.join('./dmin_image/', path.split('/')[-1])
	#temp = (1+dist_min*2)**2
	#print(np.max(model_Rx)) 
	#io.imsave(save_path,(model_Rx-1)/5)

# range of T is [2,256]
model_Tx = np.ones((img_height,img_width))



def compute_parameter_Tx(mask):
	global model_Tx,last_mask

	multi = batch_max(dist_min_lt_array,dist_min_st_array)*model_Vx*0.01
	multi = np.where(multi<0.004,0.004,multi)
	dev = (model_Vx)/batch_max(dist_min_lt_array,dist_min_st_array)

	model_Tx += np.where(mask==False,0.5/multi,-1*dev)
	#print(np.max(0.1/multi),np.min(0.1/multi))					
	#stable_reg = np.where(batch_min(dist_min_st_array,dist_min_lt_array)<UNSTABLE_REG_THRESHOLD,True,False)

	model_Tx = np.where(model_Tx<2,2.0,model_Tx)
	model_Tx = np.where(model_Tx>256,256.0,model_Tx)
	#print(np.max(temp),np.min(temp))
	#save_path = os.path.join('./dmin_image/', path.split('/')[-1])
	#print(np.max(model_Tx))
	#io.imsave(save_path,(model_Tx-1)/255)
	#img_Blur=cv2.blur(model_Vx/50,(3,3))
	#io.imsave(save_path,temp)


def update_parameter(mask,rgb_values,lbsp_values):

	global last_mask,last_rgb_values,last_lbsp_values

	compute_parameter_Vx(mask)
	compute_parameter_Rx()
	compute_parameter_Tx(mask)

	last_mask = mask.copy()
	last_rgb_values = rgb_values.copy()
	last_lbsp_values = lbsp_values.copy()
	



	

#-----------------------------------------------------------------------------------------

def debug_one_pixel_print(h=0,w=0):
	frame = {"RGB":bg_samples[h,w,:,0],
			"LBSP":bg_samples[h,w,:,1]}
	cell = DataFrame(frame,index=list(range(0,NUM_SAMPLE)))
	print(cell)


bg_tree = []
dy_tree = []
way = []
bg_car = []

#@timer
def ones_iteration(current_image,T_subsamples):
	global CURRENT_FRAME

	rgb_values  = compute_RGB_values(current_image)
	lbsp_values = compute_LBSP_values(rgb_values)
	mask = update_D_MIN_and_matching_BG_by_GPU(current_image,rgb_values,lbsp_values)
	update_samples(mask,T_subsamples,rgb_values,lbsp_values)
	#test = segmentation_image(mask)
	update_parameter(mask,rgb_values,lbsp_values)
	'''
	bg_tree.append(rgb_values[201,156])
	dy_tree.append(rgb_values[37,471])
	way.append(rgb_values[417,602])
	bg_car.append(rgb_values[403,105])
	'''
	#return test#mask




if __name__ == '__main__':
	init_bg_samples()
	subsamples = np.full((img_height,img_width),1)

	for path in img_paths:
		CURRENT_FRAME += 1
		if CURRENT_FRAME <100:
			continue
		print(path,"-----------------------------#")
		image = read_one_image(path)
		
		ones_iteration(image,subsamples)

		if CURRENT_FRAME >= 500:
			break
		#D_MIN_path = os.path.join('./D_MIN_image/', path.split('/')[-1])
		#io.imsave(D_MIN_path,dist/16)
	sio.savemat('model_Vx.mat', {"model_Vx":model_Tx})

#	hist1=np.histogram(model_Vx, bins=100) 
#	plt.hist(model_Vx.flatten(), bins=100)
#	plt.show()
	'''
	print("bg_tree",bg_samples[201,156,0,0])
	print("dy_tree",bg_samples[37,471,0,0])
	print("way",bg_samples[417,602,0,0])
	print("bg_car",bg_samples[403,105,0,0])
	plt.figure("background tree")
	plt.plot(list(range(len(bg_tree))),bg_tree)
	plt.ylim(0, 255)
	plt.figure("dynamic tree")
	plt.plot(list(range(len(dy_tree))),dy_tree)
	plt.ylim(0, 255)
	plt.figure("way")
	plt.plot(list(range(len(way))),way)
	plt.ylim(0, 255)
	plt.figure("background car")
	plt.plot(list(range(len(bg_car))),bg_car)
	plt.ylim(0, 255)
	plt.show()
	'''