
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, exposure 

'''
cutting rectangle position
'''
start = (100,200)
end = (300,400)
img_origin =  io.imread('original.jpg')
img_in = io.imread('0.jpg')

def evaluation_rgb(g_origin,g_in):

	cut_origin = g_origin[start[0]:end[0],start[1]:end[1],:].copy()
	cut_in = g_in[start[0]:end[0],start[1]:end[1],:].copy()
	cut_origin.dtype = 'int8'
	cut_in.dtype = 'int8'
	dist_matrix = abs(cut_origin-cut_in)
	dist = dist_matrix.sum()/((end[0]-start[0])*(end[1]-start[1]))
	dist_matrix.dtype = 'uint8'

	plt.subplot(3,2,3)
	io.imshow(cut_origin)
	plt.subplot(3,2,4)
	io.imshow(cut_in)
	return dist_matrix, dist

def evaluation_gray(g_origin,g_in):
	
	cut_origin = g_origin[start[0]:end[0],start[1]:end[1]].copy()
	cut_in = g_in[start[0]:end[0],start[1]:end[1]].copy()
	cut_origin.dtype = 'float64'
	cut_in.dtype = 'float64'
	dist_matrix = abs(cut_origin-cut_in)
	dist = dist_matrix.sum()

	plt.subplot(3,2,3)
	io.imshow(cut_origin)
	plt.subplot(3,2,4)
	io.imshow(cut_in)
	return dist_matrix, dist


if __name__ == '__main__':

	img_origin_gray = color.rgb2gray(img_origin)
	img_in_gray = color.rgb2gray(img_in)
#	img_in_gray = img_in
	dist_matrix, dist = evaluation_gray(img_origin_gray,img_in_gray)
	print(dist)

	plt.subplot(3,2,1)
	io.imshow(img_origin_gray)
	plt.subplot(3,2,2)
	io.imshow(img_in_gray)
	plt.subplot(3,2,5)
	io.imshow(dist_matrix)
	io.show()

