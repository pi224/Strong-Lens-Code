from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam

from sklearn import metrics


def standardCompiledSimpConvNN():
	model = simpConvNN()
	optimizer = Adam(lr = .0001, decay = 5e-5)
	model.compile(optimizer=optimizer,
	loss='binary_crossentropy',
	metrics=['accuracy'])

	return model

#Best performance with Adam, default learning rate, learning rate decay of 5e-6, 25 epochs
def simpConvNN(input_shape=(64, 64, 3)):
	model = Sequential()
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), input_shape=input_shape))
	#model went from 64x64x3 to 32x32x3
	model.add(Conv2D(64, (3, 3), strides=(2,2), activation='softplus'))
	#model is now 16x16x64
	model.add(Conv2D(32, (3, 3), activation='softplus'))
	#model is now 16x16x32
	model.add(Conv2D(16, (3, 3), activation='softplus'))
	#model is now 16x16x16
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	#model is now  8x8x16
	model.add(Flatten())
	#model is now 1024 (flattened from 8x8x16)
	model.add(Dense(128, activation='softplus'))
	model.add(Dense(32, activation='softplus'))
	model.add(Dense(1, activation='sigmoid'))
	return model

def standardCompiledSimpConvNNBatchFirst():
	model = simpConvNN()
	optimizer = Adam(lr = .000005, decay = 5e-5)
	model.compile(optimizer=optimizer,
	loss='binary_crossentropy',
	metrics=['accuracy'])

	return model

def simpConvNNBatchFirst(input_shape=(64, 64, 3)):
	model = Sequential()
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), input_shape=input_shape))
	#model went from 64x64x3 to 32x32x3
	model.add(Conv2D(64, (3, 3), strides=(2,2), activation='softplus'))
	#model is now 16x16x64
	model.add(Conv2D(32, (3, 3), activation='softplus'))
	#model is now 16x16x32
	model.add(Conv2D(16, (3, 3), activation='softplus'))
	#model is now 16x16x16
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	#model is now  8x8x16
	model.add(Flatten())
	#model is now 1024 (flattened from 8x8x16)
	model.add(Dense(128, activation='softplus'))
	model.add(Dense(32, activation='softplus'))
	model.add(Dense(1, activation='sigmoid'))
	return model

def simpConvNNSTN(input_shape=(64,64,3)):
    model = Sequential()
    model.add(SpatialTransformer(localization_net=locnet,
                             downsample_factor=3, input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), input_shape=input_shape))
	#model went from 64x64x3 to 32x32x3
	model.add(Conv2D(64, (3, 3), strides=(2,2), activation='softplus'))
	#model is now 16x16x64
	model.add(Conv2D(32, (3, 3), activation='softplus'))
	#model is now 16x16x32
	model.add(Conv2D(16, (3, 3), activation='softplus'))
	#model is now 16x16x16
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	#model is now  8x8x16
	model.add(Flatten())
	#model is now 1024 (flattened from 8x8x16)
	model.add(Dense(128, activation='softplus'))
	model.add(Dense(32, activation='softplus'))
	model.add(Dense(1, activation='sigmoid'))
	return model
