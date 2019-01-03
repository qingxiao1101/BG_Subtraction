#!/bin/bash

#category="baseline/"
#videos="office/"
#videos="highway/"

category='nightVideos/'
videos='winterStreet/'

file_path="./dataset/${category}${videos}input/"
#save_path="./binaryImage/${category}${videos}"
save_path="./save_temp_comp/"


#if folder save_image/ existent,then remove it
if [ ! -d ${save_path} ]; then
	mkdir $save_path
	echo "mkdir $save_path ..."
else
	rm  ${save_path}*
	echo "remove all image in $save_path ..."
fi

source activate py3
echo "change to conda python3 environment..."


echo "run main.py..."
python main.py $file_path $save_path

#echo "exec measure..."
#cd ../F1_Measure/
#python3 processOneScene.py
#cd ../WeberStochasticUpdate/
#source deactivate 
#echo "quit conda python3 environment..."

#source ~/.bashrc
#echo "execute 'source ~/.bashrc' "
