#!/usr/bin/env python3
import os
import numpy as np 
from pylab import imshow, show
from time import time
from numba import autojit, jit, cuda
from numpy import math


Tr = 0.3
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

'''
	- claculating one pixel`s lbsp value
	- using gray image for LBSP
'''
@jit
def lbsp_pixel(img,base_w,base_h):
		height,width = img.shape
		lbsp_value = 0
		#run_idx = 15
		for bias_w in (range(-2,3)):
			for bias_h in range(-2,3):
				if abs(bias_w)+abs(bias_h) == 3 or abs(bias_w)+abs(bias_h) == 0 :
					pass
				else:
					coordinate_w = bias_w+base_w
					coordinate_h = bias_h+base_h
					flag = 0
					if coordinate_w >= 0 and coordinate_h >= 0 and coordinate_w < width and coordinate_h < height :
						#pay attention here, type img is uint8, if no converting to float, will come some unknown problem
						if abs(np.float(img[base_h,base_w])-np.float(img[coordinate_h,coordinate_w])) <= np.float(img[base_h,base_w])*Tr :
							flag = 1
					#lbsp_value = lbsp_value + flag*(2**run_idx)
					#run_idx = run_idx - 1
					lbsp_value = (lbsp_value << 1) + flag				
		return lbsp_value

def debug_lbsp_pixel(img,base_x,base_y):
		height,width = img.shape
		lbsp_value = 0
		#run_idx = 15
		bins = []
		for bias_x in (range(-2,3)):
			for bias_y in range(-2,3):
				if abs(bias_x)+abs(bias_y) == 3 or abs(bias_x)+abs(bias_y) == 0 :
					pass
				else:
					coordinate_x = bias_x+base_x
					coordinate_y = bias_y+base_y
					flag = 0
					if coordinate_x >= 0 and coordinate_y >= 0 and coordinate_x < width and coordinate_y < height :
						if abs(img[base_y,base_x]-img[coordinate_y,coordinate_x]) <= img[base_y,base_x]*Tr :
							flag = 1
					#lbsp_value = lbsp_value + flag*(2**run_idx)
					#run_idx = run_idx - 1
					lbsp_value = (lbsp_value << 1) + flag	
					bins.append(flag)
		print("value:",lbsp_value,"bins:",bins)			
		return lbsp_value

'''
	- claculating one image`s lbsp values
	- lbsp values save in parameter values
'''
@timer
def lbsp_image_normal(img,values):
		(height, width) = img.shape
		for w in range(width):
			for h in range(height):
				values[h,w] = lbsp_pixel(img,w,h)


lbsp_pixel_for_gpu = cuda.jit(device=True)(lbsp_pixel)
@cuda.jit
def lbsp_gpu_kernel(img,values):
	height = img.shape[0]
	width = img.shape[1]

	startX, startY = cuda.grid(2)
	gridX = cuda.gridDim.x * cuda.blockDim.x;
	gridY = cuda.gridDim.y * cuda.blockDim.y;
	for x in range(startX, width, gridX):
		for y in range(startY, height, gridY): 
			values[y,x] = lbsp_pixel_for_gpu(img,x,y)
			#values[y,x] = lbsp_pixel(img,x,y)

@timer
def lbsp_image_gpu(img,values):
	lbsp_gpu_kernel[GRIDDIM, BLOCKDIM](img,values)

	

"""
if __name__ == '__main__':

	np.random.seed(0)
	image = np.random.randint(1,10,(10,10))
	lbsp_values = np.zeros(image.shape,dtype=np.uint16)

	#lbsp_image_normal(image,lbsp_values)
	lbsp_image_gpu(image,lbsp_values)
"""




















"""
	def compute_one_pixel_lbsp_value(img,coor_x,coor_y,tr = Tr):
		width, height = img.shape
		it = [x for x in list(range(-2,3)) ]
		neighbor_coordinates = [(x[0]+coor_x,x[1]+coor_y) for x in itertools.product(it,it) 
								if abs(x[0]*x[1])!=2 and (abs(x[0])+abs(x[1])!=0)]

		neighbor_values = [img[coordinate] if coordinate[0]>=0 and coordinate[1]>=0 
							and coordinate[0]<width and coordinate[1]<height
							else 0 for coordinate in neighbor_coordinates]
		bins = [1 if abs(img[coor_x,coor_y]-value)<= tr*img[coor_x,coor_y] else 0
				for value in neighbor_values]
		lbsp_value = reduce(lambda x,y: x*2+y,bins)
		return lbsp_value

	def compute_one_pixel_lbsp_value_2(img,coor_x,coor_y,tr = Tr):
		width, height = img.shape
		def func(coordinate):
			if coordinate[0]>=0 and coordinate[1]>=0 and coordinate[0]<width and coordinate[1]<height:
				if abs(img[coor_x,coor_y]-img[coordinate])<= tr*img[coor_x,coor_y]:
					return 1
				else:
					return 0
			else:
				return 0

		it = [x for x in list(range(-2,3)) ]
		neighbor_coordinates = [(x[0]+coor_x,x[1]+coor_y) for x in itertools.product(it,it) 
								if abs(x[0]*x[1])!=2 and (abs(x[0])+abs(x[1])!=0)]
		bins = list(map(func,neighbor_coordinates))
		lbsp_value = reduce(lambda x,y: x*2+y,bins)
		return lbsp_value

"""