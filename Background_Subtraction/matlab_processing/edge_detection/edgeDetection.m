clc
clear

Path = './test_img/';

Files=dir(fullfile(Path,'*.jpg'));
LengthFiles = length(Files);
Image = imread(strcat(Files(1).folder, strcat('/',Files(1).name)));
figure(1)
imshow(Image);

Igray = rgb2gray(Image);
thresh = [0.01, 0.17];
sigma = 2;
Iedge = edge(double(Igray),'canny',thresh,sigma);
figure(2)
imshow(Iedge);
