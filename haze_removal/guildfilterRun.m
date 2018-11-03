
clc
clear;

I = imread('origin_gray.jpg');
P = imread('trassimion.jpg');



I = double(I)/255.0;
P = double(P)/255.0;
I(1:10,1:10)
nhoodSize = 60;
smoothValue  = 0.001*diff(getrangefromclass(I)).^2;

%result = imguidedfilter(P,I,'NeighborhoodSize', nhoodSize, 'DegreeOfSmoothing',smoothValue);
result = guidedfilter(I,P, 30, 0.001);
result = uint8(result*255);

%subplot(1,2,1)
%imshow(I_tra)
%title("rough transmission image")
%subplot(1,2,2)
%imshow(result)
%title("transmission image by using guidefilter")

