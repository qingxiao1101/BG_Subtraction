#!/usr/bin/env python3

#FILE_PATH = './test_input/input_fall/'
#FILE_PATH = './test_input/input_office/'
#FILE_PATH = './test_input/input_highway/'
#/input_winterstreet/

#file_path='./dataset/nightVideos/winterStreet/input/'
#save_path='./save_no_update/'
number_samples=25
attribute=4                #rgb and weight features
match_threshold=2
learning_rate_st=0.04
learning_rate_lt=0.01
unstable_reg_threshold=0.1
Vx_INCR=1.0
Vx_DECR= -0.1
Tx_upper = 255.0
Tx_lower = 2.0
weber_rate_threshold = 0.32
init_threshold=8
max_num_image = 5000
weight_upper = 10000
weight_lower = 1
weight_incr = 0.1
weight_decr = 1.0
model_Rx_upper = 5.0
model_Rx_lower = 1.0
motion_area_threshold = 0.12
a=1 # for saturation in HSV

using_gpu=True

#debug variable
if_stochastic_update = False
if_update_neighboor = False
  
