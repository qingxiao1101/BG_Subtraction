import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, exposure 



'''calculating mean intensity of one gray image
def mean_intensity(image):
	return np.mean(image)
'''
def show_hist(image):

	hist1 = np.histogram(image, bins=256)
#	img = exposure.equalize_hist(image)
	plt.hist(image.flatten(),bins=20)
	plt.show()

if __name__ == '__main__':

	img_bg_origin =  io.imread('background.jpg')
	img_in_origin = io.imread('input.jpg')

	img_bg_gray = color.rgb2gray(img_bg_origin)
	img_in_gray = color.rgb2gray(img_in_origin)
	row, column = img_bg_gray.shape
	subtraction = np.zeros(img_bg_gray.shape,dtype = img_bg_gray.dtype)
	bi_subtraction = np.zeros(img_bg_gray.shape,dtype = img_bg_gray.dtype)
	subtraction = abs(img_bg_gray - img_in_gray)
	
#	subtraction[subtraction <= 0.1] = 0
#	bi_subtraction = subtraction*1.5
#	bi_subtraction[bi_subtraction > 1] = 1
#	show_hist(bi_subtraction)

#	mean_intensity = np.mean(subtraction)
#	bi_subtraction[subtraction > 0.15] = 1
#	bi_subtraction[subtraction <= 0.15] = 0

	plt.subplot(3,1,1)
	io.imshow(img_in_gray)

	plt.subplot(3,1,2)
	io.imshow(subtraction)

	plt.subplot(3,1,3)
	io.imshow(bi_subtraction)
	io.show()

