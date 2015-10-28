import numpy as np
import os

datasets_dir = 'data/'

def mnist(ntrain=50000,nvalidation=10000,ntest=10000):
	data_dir = os.path.join(datasets_dir,'')

    #load dataset for train + validation
    # get 60000 images from train
    # cut and divide 50000 images for train
    # and the last 10000 images for validation
	fd = open(os.path.join(data_dir,'train-images.idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	
	#train images
	trX = loaded[16:(50000*784)+16].reshape((50000,28*28)).astype(float)

    # validation images
	vlX = loaded[(50000*784)+16:].reshape((10000,28*28)).astype(float)
	
	# get 60000 labels from train
    # cut and divide 50000 labels for train
    # and the last 10000 labels for validation - same from images
	fd = open(os.path.join(data_dir,'train-labels.idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)

	#train labels
	trY = loaded[8:50008].reshape((50000))

	#validation labels
	vlY = loaded[50008:].reshape((10000))

	# load dataset for test
	# get 10000 images from train
	fd = open(os.path.join(data_dir,'t10k-images.idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teX = loaded[16:].reshape((10000,28*28)).astype(float)

	# test label
	fd = open(os.path.join(data_dir,'t10k-labels.idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teY = loaded[8:].reshape((10000))

	# normalization image {0,1} PB
	trX = trX/255.
	vlX = vlX/255.
	teX = teX/255.

	#train 
	trX = trX[:ntrain]
	trY = trY[:ntrain]

	#validation 
	trX = vlX[:nvalidation]
	trY = vlY[:nvalidation]

	#test
	teX = teX[:ntest]
	teY = teY[:ntest]

	return trX, teX, vlX, vlY, trY, teY