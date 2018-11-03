#!/bin/bash

echo "running removeCache.sh..."
bg_autoencoder_path='get_background_based_autoencoder/'

cd $bg_autoencoder_path 

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

cd "save_img_tmp/" && rm *
echo "remove save_img_tmp/*"

exit