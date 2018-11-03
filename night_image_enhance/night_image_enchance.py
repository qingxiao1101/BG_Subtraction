
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from haze_removal.haze_removal import haze_removal

test = haze_removal('/home/xiaoqing/Git/night_image_enhance/input.jpg')
io.imshow(test)
io.show()