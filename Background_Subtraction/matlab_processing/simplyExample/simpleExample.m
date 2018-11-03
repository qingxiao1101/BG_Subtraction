clc
clear;

background_origin = imread('background.jpg');
input_origin = imread('input.jpg');

BG_gray = rgb2gray(background_origin);
IN_gray = rgb2gray(input_origin);

Subtraction = abs(BG_gray-IN_gray);
Subtraction(Subtraction>30) = 255;

subplot(3,1,1);
imshow(BG_gray);

subplot(3,1,2);
imshow(IN_gray);

subplot(3,1,3);
imshow(Subtraction);
