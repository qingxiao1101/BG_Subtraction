
import numpy as np
from skimage import io 
import cv2


def __boxfilter(I,rad):
    N = np.ones(I.shape)
    kernel = np.ones((rad*2+1,rad*2+1),np.float32)#/(rad*rad)
    N = cv2.filter2D(I,-1,kernel)
    return N

'''
   - guidance image: I (should be a gray-scale/single channel image)
   - filtering input image: p (should be a gray-scale/single channel image)
   - local window radius: r
   - regularization parameter: eps
'''
def guidefilter(I,P,rads,eps):

    N = __boxfilter(np.ones(np.shape(I)),rads)
    
    meanI = __boxfilter(I,rads) / N
    meanP = __boxfilter(P,rads) / N
    meanIP = __boxfilter(I*P,rads) /N
    covIP = meanIP - meanI * meanP
    meanII = __boxfilter(I*I,rads) /N
    varI = meanII - meanI*meanI
    a = covIP / (varI+eps)
    b = meanP - a*meanI
    meanA = __boxfilter(a,rads) /N
    meanB = __boxfilter(b,rads) /N
    res = meanA * I + meanB
    return res


if __name__ == '__main__':
    I = io.imread('origin_gray.jpg')
    P = io.imread('trassimion.jpg')
    I = I/255.0
    P = P/255.0

    print(np.max(P))
    print(np.min(P))
    test = guidefilter(I,P,60,0.001)
    io.imshow(test)
    io.show()
