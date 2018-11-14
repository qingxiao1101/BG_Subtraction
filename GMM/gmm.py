'''
handling gray image
'''

'''
formel of data:
K 	[mu,	delta,	weight]
k1   ...	...		...
k2
k3
k4
k5	...		...		...
'''
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from pandas import Series,DataFrame
import pickle
import datetime

import os
import sys
from model_structure_ndarray import GaussianModel

alpha = 0.1
file_path = './input_image/'

def read_image_paths():
	img_names = [x for x in os.listdir(file_path) if x.split('.')[-1] in 'jpg|png']
	img_names = sorted(img_names, key = lambda x: int(x.split('.')[-2][2:]))
	img_paths = [os.path.join(file_path, x) for x in img_names]
	img = io.imread(img_paths[0], as_gray=True)
	return img_paths, img.shape

def read_first_image():
	file_list = os.listdir(file_path)
	first_image_path = file_path+file_list[0]
	img = io.imread(first_image_path,as_gray=True)

	if(isinstance (img[0,0],float)):
		img = img*255
		
	return img


def gaussian_kernel(x,mu,delta):
	return (1/(np.sqrt(2*3.14159)*delta))*np.exp(-((x-mu)**2)/(2*delta**2))


'''delete repeat element using key=(a,b)'''
def repeated_del(a,b,c):
	d = zip(a,b)
	dic = dict(zip(d,c))

	List = np.array(dic.items())	
	cc = List[:,1]
	tuples = List[:,0]
	aa,bb = zip(*tuples)
	
	return np.array(aa),np.array(bb),cc

def f1(model):
	Weight = model.get_value(col = 'weight')  #Weight(width,height,K)
	temp = np.min(Weight,axis=2)  #normal direction minimum
	t1 = np.zeros(Weight.shape)
	for k in range(Weight.shape[2]):
		t1[:,:,k] = temp
	position = np.where(Weight==t1) 
	'''position of individual(p2) with least weight in each pixel(p0,p1)'''
	p0,p1,p2 = repeated_del(position[0],position[1],position[2])
	return p0,p1,p2

def individual_sorted(model):
	Weight = model.get_value(col = 'weight')
	Delta = model.get_value(col = 'delta')
	Rank = Weight/Delta
	model.set_value(Rank,col='rank')
	individuals = model.get_value().copy()
	w,h,k,a = individuals.shape
	for i in range(w):
		for j in range(h):
			tmp = individuals[i,j]
			'''inverse sorted according the last column(namely 'rank')'''
			tmp = tmp[np.lexsort(-tmp.T)]
			individuals[i,j] = tmp
	model.model = individuals

	muster = np.zeros((w,h,k))
	for i in range(k):
		muster[:,:,i] = i+1

	model.set_value(muster,col='rank')



'''
	- replace the individual which with least weight if non matching 
'''
def update_k(model,mask,img_cur):
	#po = f1(model)
	Weight_all = model.get_value(col = 'weight') 
	weight_min = np.min(Weight_all,axis=2)
	position = np.argmin(Weight_all,axis=2)
	X,Y,K = Weight_all.shape
	xx = np.arange(0,X).reshape(X,1)
	for i in range(Y-1):
		t = np.arange(0,X).reshape(X,1)
		xx = np.hstack((xx,t))

	yy = np.arange(0,Y)
	for i in range(X-1):
		t = np.arange(0,Y)
		yy = np.vstack((yy,t))

	Mu = model.model[xx,yy,position,0].copy()
	Delta = model.model[xx,yy,position,1].copy()
	Weight = model.model[xx,yy,position,2].copy()
	'''- any(axis=0): all of k in mask is false'''
	Mu_cur = np.where(mask.any(axis=0)==False, img_cur, Mu)
	Delta_cur = np.where(mask.any(axis=0)==False, 15, Delta)
	Weight_cur = np.where(mask.any(axis=0)==False, weight_min+0.05, Weight)
	model.model[xx,yy,position,0] = Mu_cur
	model.model[xx,yy,position,1] = Delta_cur
	model.model[xx,yy,position,2] = Weight_cur


def update_weight(model,mask,k):

	Weight = model.get_value(row=k,col='weight')
	Weight = np.where(mask==True,(1-alpha)*Weight+1*alpha,(1-alpha)*Weight+0*alpha)
	model.set_value(Weight,row=k,col='weight')

'''
def update_mu_delta(model,img_current,mask,k):

	Mu_pre = model.get_value(row=k,col='mu')
	Delta_pre = model.get_value(row=k,col='delta')
	rho = alpha*gaussian_kernel(img_current,Mu_pre,Delta_pre)
	Mu_cur = np.where(mask[k]==True, Mu_pre + (img_current-Mu_pre)*rho,Mu_pre)
	Delta_cur = np.where(mask[k]==True, Delta_pre + (img_current-Mu_pre)**2 * rho, Delta_pre)
	model.set_value(Mu_cur,row=k,col='mu')
	model.set_value(Delta_cur,row=k,col='delta')
'''

def update_mu_delta(model,img_current,mask,k):

	Mu_pre = model.get_value(row=k,col='mu')
	Delta_pre = model.get_value(row=k,col='delta')
	rho = alpha*gaussian_kernel(img_current,Mu_pre,Delta_pre)
	Mu_cur = np.where(mask==True, Mu_pre + (img_current-Mu_pre)*rho,Mu_pre)
	Delta_cur = np.where(mask==True, Delta_pre + (img_current-Mu_pre)**2 * rho, Delta_pre)
	model.set_value(Mu_cur,row=k,col='mu')
	model.set_value(Delta_cur,row=k,col='delta')

def debug_one_pixel(model,mask,img_cur,xx,yy):
	print("(%d,%d) pixel value: %f"%(xx,yy,img_cur[xx,yy]))
	print(mask[:,xx,yy])
	model.debug_print(xx,yy)
	

def train_GMM(model,img_current):
	'''if img value in k1-k5 with 2.5 times delta'''
	#step 1: comparison with current image
	Width,Height,K,Attribute = model.model.shape
	mask = np.full((K,Width,Height),False)
	for k in range(K):
		previous = model.get_value(row=k,col='mu')
		t1 = abs(img_current - previous)
		t2 = 2.5 * model.get_value(row=k,col='delta')
		'''False: value over 2.5 times delta'''
		mask[k] = np.where(t1<t2,True,False)
		#print("k%d previous: %f,t1: %f,t2: %f"%(k,previous[100,200],t1[100,200],t2[100,200]))

	#step 2: updatting weight
		update_weight(model,mask[k],k)
	#step 3: updatting delta and mu
		update_mu_delta(model,img_current,mask[k],k)


		#np.where(mask[k]==True,individual_matching(model,img_current,k),individual_no_matching(model,k))
	model.weight_normalization()

	#step 4: if in 5 individual non matching,then executing update_k 
	update_k(model,mask,img_current)
	model.weight_normalization()		

	#step 5: individual sorted...
	#individual_sorted(model)
	debug_one_pixel(model,mask,img_current,200,100)

'''
	setting the mu of k1 first-image`s gray value, weight as 1, each variance is 10
	- model : gaussian mixture model
	- f_img : the first image 
'''
def preset_model(model,f_img):
	Width,Height,K,Attribute = model.model.shape
	model.set_value(f_img,row='k1',col='mu')
	model.set_value(np.ones(f_img.shape),row='k1',col='weight')
	model.set_value(np.full((Width,Height,K),10),col='delta')

def run(model,img_paths):
	for path in img_paths:
		img = io.imread(path,as_gray=True) * 255.0
		print("calculating %s ..."%(path))
		train_GMM(model,img)
		print(" ")


if __name__ == '__main__':
	img_paths,(row,col) = read_image_paths()
	f_image = read_first_image()
	model = GaussianModel()
	preset_model(model,f_image)
	run(model,img_paths)
	#train_GMM(model,f_img)
	individual_sorted(model)

	model.debug_print(200,100)

	
