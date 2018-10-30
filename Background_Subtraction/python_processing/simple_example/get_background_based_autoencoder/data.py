import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from skimage import io, transform

class Data():
    """
    Prepare data for training and testing 
    """
    def __init__(self):   #constructor
        pass
    
    def get_image(self, file_path = './train_image'):
        img_names = [x for x in os.listdir(file_path) if x.split('.')[-1] in 'jpg|png']
        img_names = sorted(img_names, key = lambda x: int(x.split('.')[-2][2:]))
        img_paths = [os.path.join(file_path, x) for x in img_names]
        image = io.imread(img_paths[1], as_grey = True)
        for img_path in img_paths:
            img = io.imread(img_path, as_grey = True)    #shape img: 420*624 print(img.shape) 
#            img = transform.resize(img, (300, 480))    #caution the difference between resize and reshape
            if img_paths.index(img_path) == 0: #reading the first image
                img_matrix = img[np.newaxis, :]
            else:
                img_matrix = np.concatenate((img_matrix, img[np.newaxis,:]), axis = 0)

        img_info = []
        row,col = image.shape
        img_info.append(row)
        img_info.append(col)
        img_info.append(len(img_paths))
        return img_matrix, img_info  # img_matrix: n*300*480 (n number of images)

if __name__ == '__main__':
    data = Data()
    data.get_image()
