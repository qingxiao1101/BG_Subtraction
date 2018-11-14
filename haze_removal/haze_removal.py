
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, exposure 
import skimage.filters.rank as sfr
from skimage.morphology import square #rectangular filter-kernel
#from skimage.morphology import disk 	#circular filter-kernel

import cv2

import guide_filter as gf




'''
	step 1:
	getting dark_channel by using min Filter to min_channel
	Min_channel defines as minimum of {r,g,b} in all pixel
'''
def __calculate_dark_channel(image, rads):
	(row, col, channel) = image.shape
	Min_channel = np.zeros((row,col),'uint8')
	Min_channel = image.min(axis=2) 
	dark_channel = sfr.minimum(Min_channel,square(rads))

	return dark_channel

'''
	- image : original image
	- dark : dark_channel image
	- limit : limit of A 
'''
def __calculate_atmospheric_light(image, dark, limit = 200):
	hist = np.histogram(dark, bins=100)
	position = np.where(dark>=hist[1][-2])
	A = np.zeros(3,dtype = 'double')
	for i in range(image.shape[2]):
		A[i] = np.mean(image[position[0],position[1],i]) 
		if A[i] >= limit:
			A[i] = limit

	return A

'''
	calculating transmition t(x) with guide filter
	- image : orignal image
	- A : atmospheric light
	- rads : filter radius (default 15)
	- w : modification coefficient
'''
def __calculate_transmition_image(image, A, rads, eps = 0.001, w = 0.95):

	Iy = np.zeros(image.shape,'double')
	for i in range(image.shape[2]):
		Iy[:,:,i] = image[:,:,i]/A[i]
	Imin_c = Iy.min(axis=2)
	pos = np.where(Imin_c > 1)
	Imin_c[pos[0],pos[1]] = 1 / Imin_c[pos[0],pos[1]]
	Imin_filter = sfr.minimum(Imin_c,square(rads))
	rough_transmition = 1 - w * (Imin_filter/255.0)
	origin_gray = color.rgb2gray(image)
	transmition_image = gf.guidefilter(origin_gray,rough_transmition,rads*4,eps)
#	io.imsave('../transmition.jpg',transmition_image)

	return transmition_image


def __recover_no_foggy_image(image, transmition, A, gamma = 0.588):
	T = np.zeros((transmition.shape[0],transmition.shape[1],2))
	T[:,:,0] = np.full(transmition.shape,0.1)
	T[:,:,1] = transmition
	T_max = T.max(axis=2)

	J_img = np.zeros(image.shape)
	A_scale = A/250.0
	img_scale = image/255.0
	for i in range(image.shape[2]):
		J_img[:,:,i] =A_scale[i] - (A_scale[i] - img_scale[:,:,i])/T_max
		p1 = np.where(J_img[:,:,i]>1)
		J_img[p1[0],p1[1],i] = 1
		p2 = np.where(J_img[:,:,i]<0)
		J_img[p2[0],p2[1],i] = 0		

	J_img_gamma = np.power(J_img/float(np.max(J_img)), gamma)

	return J_img_gamma

def haze_removal(original_img, rads = 5):
	original = cv2.imread(original_img)
	dark_channel = __calculate_dark_channel(original, rads)
	A = __calculate_atmospheric_light(original, dark_channel)
	transmition = __calculate_transmition_image(original,A,rads)
	J_img = __recover_no_foggy_image(original,transmition,A)

	return J_img

'''
if __name__ == '__main__':

	no_foggy_img = haze_removal('/home/xiaoqing/Project_git/night_image_enhance/input.jpg')

	plt.subplot(2,1,1)
	io.imshow(original_img)
	plt.subplot(2,1,2)
	io.imshow(no_foggy_img)
	io.show()
'''


