'''
This is a place for us to put all our preprocessing functions
gaussian filter is in scipy.ndimage.filters
'''

import numpy
import scipy.fftpack as fp

'''
images input should be numpy nd array
function came from this stackoverflow post. Dubious reliability,
but I think worth a try
'''
def fft(images):
	return numpy.asarray([fp.rfft(fp.rfft(image, axis=0), axis=1) for
			image in images])

#zca whitening. epsilon is there to prevent division by 0
def ZCA(images, epsilon):
	#flattening images first
	images = numpy.asarray([image.flatten('F') for image in images])
	"""
	Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
	INPUT:  X: [M x N] matrix.
		Rows: Variables
		Columns: Observations
	OUTPUT: ZCAMatrix: [M x M] matrix
	"""
	# Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
	sigma = numpy.cov(X, rowvar=True) # [M x M]
	# Singular Value Decomposition. X = U * numpy.diag(S) * V
	U,S,V = numpy.linalg.svd(sigma)
		# U: [M x M] eigenvectors of sigma.
		# S: [M x 1] eigenvalues of sigma.
		# V: [M x M] transpose of U
	# ZCA Whitening matrix: U * Lambda * U'
	ZCAMatrix = numpy.dot(U, numpy.dot(numpy.diag(1.0/numpy.sqrt(S + epsilon)), U.T)) # [M x M]
	
	ZCAMatrix = numpy.asarray([image.reshape((64, 64, 3), order='F')
			for image in ZCAMatrix])
	return ZCAMatrix, U.T #U.T is the projection matrix. 