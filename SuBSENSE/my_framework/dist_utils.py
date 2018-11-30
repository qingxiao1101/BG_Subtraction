#!/usr/bin/env python3

import numpy as np 
from numba import vectorize, float64, int64,float32,int32,uint16
from numba import autojit, jit, cuda
from time import time

def timer(func):
    def deco(*args, **kwargs):  
        start = time()
        res = func(*args, **kwargs)
        stop = time()
        print('function (%s) cost %f seconds' %(func.__name__,stop-start))
        return res 
    return deco

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



