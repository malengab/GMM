#import tensorflow as tf
from __future__ import division
import numpy as np
import math, random
import matplotlib.pyplot as plt
from numpy.matlib import zeros
import pdb

#%matplotlib inline

def generate_data(N,bump_max):
	fction = lambda x,y,a: a*np.exp(-10*(y-x)**2)
	d,eps = 3, 0.1; 		# domain length/2
#	N = 100	# nr of pts
	rep = np.random.randint(bump_max)	# how many bumps	
	x = np.linspace(-d, d, N)	# domain pts
	hstep = 2*d/(N-1);	# steplength
	y = fction(x,2-4*np.random.random(),1)	# initialization
	for r in range(0,rep-1):
		cen = 2-4*np.random.random()
		y = y + fction(x,cen,1)		# solution superposed
	yder = np.array((y[1:]-y[0:len(y)-1]), dtype='f')/hstep	# function derivative
	ydata = np.concatenate((y,yder), axis=0)
#	pdb.set_trace()
	rep_one_hot = [0]*bump_max;	# save label as [0,... 0,1,0,... 0]
	rep_one_hot[rep] = 1;
#	plt.figure(1)
##	print(rep_one_hot)
#	plt.scatter(x[0:], y, c='green', label='train')
##	plt.legend()
#	plt.show()
#	plt.close()
#	plt.figure(2)
##	print(rep_one_hot)
#	plt.scatter(x[1:], yder, c='green', label='train')
#		
	return y,rep_one_hot

def create_feature_sets_and_labels(repeat,test_size,bump_max):
	features = []
	for ii in range(repeat):
		features += [generate_data(200,bump_max)]
	random.shuffle(features)
	features = np.array(features)

	testing_size = int(test_size*len(features))
#	feature_x = features[0::2]	# array of values
#	feature_y = features[1::2]	# labels
#	pdb.set_trace()

	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])
	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])

	return train_x,train_y,test_x,test_y

#if __name__ == '__main__':
#	train_x,train_y,test_x,test_y = create_feature_sets_and_labels(100,.8,10)#('/path/to/pos.txt','/path/to/neg.txt')
#	print(train_x,train_y,test_x,test_y)
