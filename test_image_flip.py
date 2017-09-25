import keras
import numpy
import utils
import metrics
import models_preprocessing


from keras.preprocessing.image import ImageDataGenerator


auroc = metrics.auroc

reg_convnet = models_preprocessing.compiledRegularizedConvnet 

data = numpy.load('imadjust.npy')

labels = numpy.load('classification.npy')

generator = ImageDataGenerator(featurewise_center=True,
	featurewise_std_normalization=True, 
	rotation_range=90, 
	horizontal_flip=False)


utils.epoch_curve_generator(reg_convnet, data, labels, generator, 32, 0.3, 30, auroc)
