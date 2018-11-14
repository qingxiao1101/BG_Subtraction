
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import os
import sys
#parentdir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,'../')
from haze_removal.haze_removal import haze_removal

img_path = '/home/xiaoqing/Project_git/night_image_enhance/input.jpg'



test = haze_removal()
io.imshow(test)
io.show()

