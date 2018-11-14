import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from pandas import Series,DataFrame
import pickle
import datetime

import os
import sys
'''
formel of data:
K 	[mu,	delta,	weight]
k1   ...	...		...
k2
k3
k4
k5	...		...		...
'''
K = 5
ATTRIBUTE = 4
file_path = './input_image/'
file_list = os.listdir(file_path)
first_image_path = file_path+file_list[0]
ROW,COL = io.imread(first_image_path,as_gray=True).shape
M = {'k1':0,'k2':1,'k3':2,'k4':3,'k5':4,'mu':0,'delta':1,'weight':2,'rank':3}

class GaussianModel():

	def __init__(self):   #constructor
		self.model = np.zeros((ROW,COL,K,ATTRIBUTE),dtype = 'float32')
	'''
		- xx,yy : position of individual
		- row,col: value in individual
		- field : setting value in (row,col)
	'''
	def set_value(self,field,row=-1,col=-1,xx=-1,yy=-1):
		if (row== -1 and col!=-1 and xx==-1 and yy==-1):
			self.model[:,:,:,col if(isinstance(col,int)) else M[col]] = field.copy()
		elif(row!= -1 and col!=-1 and xx==-1 and yy==-1):
			self.model[:,:,row if(isinstance(row,int)) else M[row],col if(isinstance(col,int)) else M[col]] = field.copy()
		elif(row!= -1 and col!=-1 and xx!=-1 and yy!=-1):
			self.model[xx,yy,row if(isinstance(row,int)) else M[row],col if(isinstance(col,int)) else M[col]] = field.copy()
		elif(row== -1 and col==-1 and xx!=-1 and yy!=-1):
			self.model[xx,yy,:,:] = field.copy()
		else:
			raise RuntimeError('setting value Error')
			

	def get_value(self,row=-1,col=-1,xx=-1,yy=-1):
		#given nothing
		if (row== -1 and col==-1 and xx==-1 and yy==-1):
			return self.model[:,:,:,:].copy()
		#given row only...
		elif(row!= -1 and col==-1 and xx==-1 and yy==-1):
			return self.model[:,:,row if(isinstance(row,int)) else M[row],:].copy()
		#given col only...
		elif(row== -1 and col!=-1 and xx==-1 and yy==-1):
			return self.model[:,:,:,col if(isinstance(col,int)) else M[col]].copy()
		#given row and col
		elif(row!= -1 and col!=-1 and xx==-1 and yy==-1):
			return self.model[:,:,row if(isinstance(row,int)) else M[row],col if(isinstance(col,int)) else M[col]].copy()
		#given row,col,xx,yy
		elif(row!= -1 and col!=-1 and xx!=-1 and yy!=-1):
			return self.model[xx,yy,row if(isinstance(row,int)) else M[row],col if(isinstance(col,int)) else M[col]].copy()
		#given xx,yy only
		elif(row== -1 and col==-1 and xx!=-1 and yy!=-1):
			return self.model[xx,yy,:,:].copy()
		#given xx,yy and col
		elif(row== -1 and col!=-1 and xx!=-1 and yy!=-1):
			return self.model[xx,yy,:,col if(isinstance(col,int)) else M[col]].copy()
		#given xx,yy and row 
		elif(row!= -1 and col==-1 and xx!=-1 and yy!=-1):
			return self.model[xx,yy,row if(isinstance(row,int)) else M[row],:].copy()	
		else:
			raise RuntimeError('getting value Error')

	def weight_normalization(self):
		result = np.sum(self.model,axis=2)[:,:,2] 	#mu-0,delta-1,weight-2
		for k in ['k1','k2','k3','k4','k5']:
			div = self.get_value(row=k,col='weight')/result
			self.set_value(div,row=k,col='weight')
				
	
	def debug_print(self,xx=0,yy=0):
		frame = {"mu":self.model[xx,yy,:,0],
				 "delta":self.model[xx,yy,:,1],
				 "weight":self.model[xx,yy,:,2],
				 "rank":self.model[xx,yy,:,3],}
		cell = DataFrame(frame,index=['k1','k2','k3','k4','k5'])
		print(cell)

'''
model = GaussianModel()
model.set_value(10,row='k1',col=2)
Weight = model.get_value(col='weight')
print(Weight.shape)
model.debug_print(4,4)
'''

