#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import time
import os
import sys
import cv2
import configuration as config
from algorithms import Model
import utils_func
import pickle

file_path = sys.argv[1]
binary_path = sys.argv[2]

bg_samples = 'bg_samples.data'
min_weight_index = 'min_weight_index.data'
def read_image_paths(filePath = file_path )->[str]: 
	img_names = [x for x in os.listdir(filePath) if x.split('.')[-1] in 'jpg|png']
	img_names = sorted(img_names, key = lambda x: int(x.split('.')[-2][2:]))
	img_paths = [os.path.join(filePath, x) for x in img_names]
	return img_paths

def read_one_image(imgPath) -> np.array: #return uint8
	img = cv2.imread(imgPath)
	return img

def timer(func):
    def deco(*args, **kwargs):  
        start = time.time()
        res = func(*args, **kwargs)
        stop = time.time()
        print('function (%s) cost %f seconds' %(func.__name__,stop-start))
        return res 
    return deco


@timer
def run(start=0, end=config.max_num_image):
	img_paths = read_image_paths(file_path)
	init_image = read_one_image(img_paths[0])
	newdim = list(init_image[...,np.newaxis].shape)
	newdim[-1]=12
	init_images = np.zeros(tuple(newdim))
	for i in range(12):
		init_images[...,i] = read_one_image(img_paths[i])

	model = Model(init_image,init_images)

	idx = 0

	for path in img_paths:
		idx += 1
		if idx < start:
			continue
		image = cv2.imread(path)
		#seg = model.lighting_check(image)
		seg = model.ones_iteration(image,path)
		#model.save_variables(binary_path,seg)
		model.save_variables(binary_path,model.min_weber_rate)
		print("processing %s ... ... ..."%(path))
		if idx > end-1:
			break
	#model.save_mat(model.debug_data)

	#f = open(bg_samples, 'wb')
	#f2 = open(min_weight_index, 'wb')
	# 将变量存储到目标文件中区
	#pickle.dump(model.bg_samples, f)
	#pickle.dump(model.min_weight_index, f2)
	# 关闭文件
	#f.close()
	#f2.close()

	#model.debug_one_pixel_print(10,10)
	#print('idx: ',model.min_weight_index[10,10])
	#print('max_weight: ',model.max_weight[10,10])


if __name__ == '__main__':
	
	run(0,100)
	#run()
	#model.debug_one_pixel_print(0,0)
	#cv2.imshow('name',first_img)
	#cv2.waitKey(0)
	#run(img_paths)



		

