# Gaussian Mixture Models
Gaussian mixture models, that is a function that can be written as a sum of Gaussian functions
Purpose: find the coefficients, variances and means of the underlying Gaussian functions given the superposed function

Files:

gb.py
Generates a set of training and test data given the number of repeats, the size of the test set (in %) and the maximal number of Gaussian elements (bump_max). Each entry then includes function values of a 1D GMM function given the number of discretization points (set to N=100 for now) over a fixed [-3,3] interval. All amplitudes are set to 1 and the number of GB bumps is chosen randomly, bound by bump_max.

GBtensorflow.py
Uses 3 hidden tensorflow layers to teach the neural network to recognize the number of GB bumps based of the function values taken from gb.py (that is, not their position or shape, just number).

GMM.py
Expectation-maximization technique (unsupervised learning) to determine the means, coefficients and variances of the GB functions given the superposition function values obtained again from gb.py (normalized to 1). The number of bumps is assumed to be known.

credits: Tensorflow and python tutorial in
https://pythonprogramming.net/tensorflow-neural-network-session-machine-learning-tutorial/?completed=/tensorflow-deep-neural-network-machine-learning-tutorial/

G. Ariel, B. Enquist, N. M. Tanushev, R. Tsai: Gaussian beam decomposition of high frequency wave fiels using expectation-maximization, J of Comp Phys, 2011.
