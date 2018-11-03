
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, exposure 
import skimage.filters.rank as sfr
from skimage.morphology import square #rectangular filter-kernel
#from skimage.morphology import disk    #circular filter-kernel

import cv2

import guide_filter as gf


w = 0.95


if __name__ == '__main__':

    img = cv2.imread("foggy2.jpg")
    (row, col, channel) = img.shape

    '''
    step 1:
    getting dark_channel by using min Filter to min_channel
    Min_channel defines as minimum of {r,g,b} in all pixel
    '''
    Min_channel = np.zeros((row,col),'uint8')
    Min_channel = img.min(axis=2) 
    dark_channel = sfr.minimum(Min_channel,square(15))

    '''
    step 2:
    calculating atmospheric light A and using limit with A<=210
    '''
    #hist = exposure.histogram(dark_channel, nbins=20)
    hist = np.histogram(dark_channel, bins=100)
    position = np.where(dark_channel>=hist[1][-2])
    A = np.zeros(3,dtype = 'double')
    for i in range(channel):
        A[i] = np.mean(img[position[0],position[1],i]) 
        if A[i] >= 210:
            A[i] = 210

    '''
    step 3:
    calculating trassimion t(x)
    '''
    Iy = np.zeros((row,col,channel),'double')
    for i in range(channel):
        Iy[:,:,i] = img[:,:,i]/A[i]
    Imin_c = Iy.min(axis=2)
    pos = np.where(Imin_c > 1)
    Imin_c[pos[0],pos[1]] = 1 / Imin_c[pos[0],pos[1]]
    Imin_filter = sfr.minimum(Imin_c,square(15))
    Trassimion = 1 - w * (Imin_filter/255.0)

#   io.imsave('../save_transmission.jpg',Trassimion)

    origin_gray = color.rgb2gray(img)
    guildfilter_image = gf.guidefilter(origin_gray,Trassimion,60,0.001)

    '''
    step 4: 
    
    '''

    T = np.zeros((guildfilter_image.shape[0],guildfilter_image.shape[1],2))
    T[:,:,0] = np.full(guildfilter_image.shape,0.1)
    T[:,:,1] = guildfilter_image
    T_max = T.max(axis=2)

    J_img = np.zeros(img.shape)
    A = A/250.0
    img_scale = img/255.0
    for i in range(channel):
        J_img[:,:,i] =A[i] - (A[i] - img_scale[:,:,i])/T_max
        p1 = np.where(J_img[:,:,i]>1)
        J_img[p1[0],p1[1],i] = 1
        p2 = np.where(J_img[:,:,i]<0)
        J_img[p2[0],p2[1],i] = 0        

    img1 = np.power(J_img/float(np.max(J_img)), 1/1.7)
    print(np.max(J_img))
    print(np.min(J_img))

    plt.subplot(3,1,1)
    io.imshow(img)
    plt.subplot(3,1,2)
    io.imshow(J_img)
    plt.subplot(3,1,3)
    io.imshow(img1)
    io.show()

