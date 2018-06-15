from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pdb
import math, random
from gb import generate_data

N = 100; 	# nr of discretization pts
bump_max = 10; 	# max nr of Gaussian components
y, nr_bumps = generate_data(N,bump_max) 	# nr of bumps is in one-hot style
y = y/sum(y)	# normalize to 1

d = 3; 		# domain length/2
x = np.linspace(-d, d, N)	# domain pts
nrb = np.nonzero(nr_bumps)	# transform from one hot to nr
nrb = nrb[0][0]
print(nrb)

#plt.figure(1)
#plt.scatter(x, y, c='green', label='orig')
#plt.legend()
#plt.show()
#plt.close()

# start with nrb random guesses for GMM means, variances, distribution components
me = [np.random.uniform(-d/2,d/2) for j in range(0,nrb)]	#
var = [.5]*nrb
psi = [1/nrb]*nrb
pij = np.zeros((nrb,N))		# initiate probabilities

# first guess
fction = lambda x,s,v: np.exp(-((s-x)**2)/2/v)	# Gaussian fction
nit = 30		# number of iterations

for j in range(nit):
	z = np.zeros(N) # initialization, zero everywhere
#	pdb.set_trace()
	########## EXPECTATION
	for r in range(0,nrb):
			g = fction(x,me[r],var[r])		# Gaussian bump
			zn = sum(g);	# normalization
			z = z + psi[r]*g/zn		# solution superposed
			pij[r] = psi[r]*g/zn	# save numerator of probablities

	for r in range(0,nrb):
		pij[r] = np.divide(pij[r],z)
#		pij = np.ones((nrb,N))

	########### MAXIMIZATION
	# iterate variables
	psinew = np.dot(pij,y)
	menew = np.divide(np.dot(pij,np.multiply(y,x)),psinew)
	varnew = np.divide(np.dot(pij,np.multiply(y,x**2)),psinew)-menew**2
#	pdb.set_trace()
	
#	print(psi, me, var, psinew, menew, varnew)
	psi, me, var = psinew, menew, varnew	# update old variables
	
plt.scatter(x, y, c='blue', label='original')
plt.scatter(x, z, c='red', label='iter')
plt.legend()
plt.show()
#	pdb.set_trace()
