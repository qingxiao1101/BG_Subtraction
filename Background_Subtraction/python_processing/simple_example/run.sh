#!/bin/bash

<<<<<<< HEAD
=======
#origin_img_path='/home/xiaoqing/Resource/dataset2014/dataset/nightVideos/winterStreet/input/'
>>>>>>> 58c5979052b627ca81e7dfef157a44d4e2258687
#get current path
current_path=$(cd `dirname $0`; pwd)

bg_autoencoder_path='get_background_based_autoencoder/'

echo "running bg_autoencoder.sh..."
cd $bg_autoencoder_path && ./bg_autoencoder.sh

exit
