#!/bin/bash

origin_img_path='/home/xiaoqing/Resource/dataset2014/dataset/nightVideos/winterStreet/input/'
#get current path
current_path=$(cd `dirname $0`; pwd)

#if folder save_image/ existent,then remove it
if [  -d "save_image/" ]; then
  rm -rf save_image/
  echo "remove save_image/..."
fi
if [  -d "test_image/" ]; then
  rm -rf test_image/
  echo "remove test_image/..."
fi
if [  -d "train_image/" ]; then
  rm -rf train_image/
  echo "remove train_image/..."
fi
if [  -d "model_save/" ]; then
  rm -rf model_save/
  echo "remove model_save/..."
fi

#creating temporal folder
mkdir save_image/ test_image/ train_image/ model_save/
echo "create save_image/, test_image/, train_image/, model_save/ folder..."

#copy the first 400 images from dataset2014 to train_image/
cp ${origin_img_path}'in0000'* train_image/
cp ${origin_img_path}'in0001'* train_image/  
<<<<<<< HEAD
#cp ${origin_img_path}'in0002'* train_image/ 
#cp ${origin_img_path}'in0003'* train_image/ 
#cp ${origin_img_path}'in0004'* train_image/ 
#cp ${origin_img_path}'in0005'* train_image/ 
cp ${origin_img_path}'in000200.jpg' train_image/
=======
cp ${origin_img_path}'in0002'* train_image/ 
cp ${origin_img_path}'in0003'* train_image/ 
cp ${origin_img_path}'in0004'* train_image/ 
cp ${origin_img_path}'in0005'* train_image/ 
cp ${origin_img_path}'in000600.jpg' train_image/
>>>>>>> 58c5979052b627ca81e7dfef157a44d4e2258687
echo "copy the first 600 images from dataset2014 to train_image/..."


#copy 10 images(number 1000-1009) from dataset2014 to train_image/
cp ${origin_img_path}'in00100'* test_image/
echo "copy 10 images(number 1000-1009) from dataset2014 to train_image/"

#run Autoencoder.py
echo "running Autoencoder.py..."
python Autoencoder.py
echo "running Autoencoder.py successful"


exit