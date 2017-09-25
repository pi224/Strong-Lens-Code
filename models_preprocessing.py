from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2


def compiledConvnet(input_shape=(101, 101, 4)):
	model = convnet(input_shape)
	optimizer = Adam(lr = .0001, decay = 5e-5)
	model.compile(optimizer=optimizer,
	loss='binary_crossentropy',
	metrics=['accuracy'])

	return model

def convnet(input_shape=(101, 101, 4)):
	model = Sequential()
	model.add(Conv2D(64, (3, 3), strides=(2,2), activation='softplus',
					input_shape=input_shape))
	model.add(Conv2D(32, (3, 3), strides=(2,2), activation='softplus'))
	model.add(Conv2D(16, (3, 3), activation='softplus'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	model.add(Flatten())

	model.add(Dense(128, activation='softplus'))
	model.add(Dense(32, activation='softplus'))
	model.add(Dense(1, activation='sigmoid'))
	return model

def compiled_maxpool_simpler_1(input_shape=(101, 101, 4)):
	model = maxpool_simpler_1(input_shape)
	optimizer = Adam(lr = .0001, decay = 5e-5)
	model.compile(optimizer=optimizer,
	loss='binary_crossentropy',
	metrics=['accuracy'])

	return model

def maxpool_simpler_1(input_shape=(101, 101, 4)):
	model = Sequential()
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), input_shape=input_shape))
	model.add(Conv2D(32, (3, 3), strides=(2,2), activation='softplus',
					input_shape=input_shape))
	model.add(Conv2D(16, (3, 3), strides=(2,2), activation='softplus'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	model.add(Flatten())

	model.add(Dense(128, activation='softplus'))
	model.add(Dense(32, activation='softplus'))
	model.add(Dense(1, activation='sigmoid'))
	return model

def compiled_maxpool_convnet(input_shape=(101, 101, 4)):
	model = convnet(input_shape)
	optimizer = Adam(lr = .0001, decay = 5e-5)
	model.compile(optimizer=optimizer,
	loss='binary_crossentropy',
	metrics=['accuracy'])

	return model

def maxpool_convnet(input_shape=(101, 101, 4)):
	model = Sequential()
	model.add(MaxPooling2D(pool_size=(4,4), strides=(4,4), input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), strides=(2,2), activation='softplus'))
	model.add(Conv2D(32, (3, 3), strides=(2,2), activation='softplus'))
	model.add(Conv2D(16, (3, 3), activation='softplus'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	model.add(Flatten())

	model.add(Dense(128, activation='softplus'))
	model.add(Dense(32, activation='softplus'))
	model.add(Dense(1, activation='sigmoid'))
	return model

def compiledRegularizedConvnet(input_shape=(101, 101, 4)):
	model = regularizedConvnet(input_shape)
	optimizer = Adam(lr = .0001, decay = 5e-5)
	model.compile(optimizer=optimizer,
	loss='binary_crossentropy',
	metrics=['accuracy'])

	return model

reg_1 = 0.3
reg_2 = 0
reg_3 = 0
def regularizedConvnet(input_shape=(101, 101, 4)):
	model = Sequential()
	model.add(Conv2D(64, (3, 3), strides=(2,2), activation='softplus',
					input_shape=input_shape))
	model.add(Conv2D(32, (3, 3), strides=(2,2), activation='softplus'))
	model.add(Conv2D(16, (3, 3), activation='softplus'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	model.add(Flatten())

	model.add(Dense(128, activation='softplus', kernel_regularizer=l2(reg_1)))
	model.add(Dense(32, activation='softplus', kernel_regularizer=l2(reg_2)))
	model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(reg_3)))
	return model
