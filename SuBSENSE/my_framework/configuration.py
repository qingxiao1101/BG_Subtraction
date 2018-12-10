#!/usr/bin/env python3

#FILE_PATH = './test_input/input_fall/'
#FILE_PATH = './test_input/input_office/'
#FILE_PATH = './test_input/input_highway/'

file_path='./test_input/input_highway/'
number_samples=50
attribute=2                #rgb and lbsp features
rgb_init_threshold=30
lbsp_init_threshold=3
match_threshold=2
learning_rate_st=0.04
learning_rate_lt=0.01
unstable_reg_threshold=0.1
Vx_INCR=1.0
Vx_DECR= -0.1
Tx_upper = 256.0
Tx_lower = 2.0
max_num_image = 5000
#using_gpu=False
using_gpu=True


