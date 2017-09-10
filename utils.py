import numpy

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)

def train(compiled_model, epochs, trainX_file, trainY_file):
	print(compiled_model.summary())

	if type(trainX_file) is str:
		trainX = numpy.load(trainX_file)
		trainY = numpy.load(trainY_file)
	else:
		trainX = trainX_file
		trainY = trainY_file
		trainX_file = 'training X'
		trainY_file = 'training Y'
	try:
        datagen  = keras.preprocessing.image.ImageDataGenerator(
                                             zca_whitening=True,
                                             zca_epsilon=1e-2,
                                             featurewise_center=True,
                                             featurewise_std_normalization=True,
                                             rotation_range=20,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             horizontal_flip=True)
        datagen.fit(trainX)
        
        compiled_model.fit_generator(datagen.flow(trainX, trainY, batch_size=32),
                    steps_per_epoch=len(trainX) / 32, epochs = epochs)
    
    except:
	
        return compiled_model

	return compiled_model

    
def cross_validate(compiled_model, epochs, trainX_file, trainY_file):
	print(compiled_model.summary())

	if type(trainX_file) is str:
		trainX = numpy.load(trainX_file)
		trainY = numpy.load(trainY_file)
	else:
		trainX = trainX_file
		trainY = trainY_file
		trainX_file = 'training X'
		trainY_file = 'training Y'
	try:
        datagen  = keras.preprocessing.image.ImageDataGenerator(
                                             zca_whitening=True,
                                             zca_epsilon=1e-2,
                                             featurewise_center=True,
                                             featurewise_std_normalization=True,
                                             rotation_range=20,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             horizontal_flip=True)
        datagen.fit(trainX)
        
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        
        cvscores = []
        
        for train, test in kfold.split(X, Y):
            # fit the model
            compiled_model.fit_generator(datagen.flow(trainX, trainY, batch_size=32),
                    steps_per_epoch=len(trainX) / 32, epochs = epochs)
            # evaluate the model
            scores = model.evaluate(X[test], Y[test], verbose=0)
            
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1] * 100)
        
        print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
        
        
    
    except:
	
        return compiled_model

	return compiled_model
    
    
    

def test(trained_model, metrics_array, *argv):
	testing_sets = [(0, 0)]

	try:
		testing_sets = pairwise(argv)
	except:
		print('an error occurred with your testing sets. Are you missing \
			some input vectors or some labels? Or maybe you paired the name \
			of a file and a numpy array together? (we do not support that yet)')
		return

	counter = 1
	for X_file, Y_file in testing_sets:
		X, Y = (0, 0)
		if type(X_file) is str:
			X = numpy.load(X_file)
			Y = numpy.load(Y_file)
		else:
			X = X_file
			Y = Y_file
			X_file = 'testing X ' + str(counter)
			Y_file = 'testing Y ' + str(counter)

        datagen  = keras.preprocessing.image.ImageDataGenerator(
                                             zca_whitening=True,
                                             zca_epsilon=1e-2,
                                             featurewise_center=True,
                                             featurewise_std_normalization=True,
                                             rotation_range=20,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             horizontal_flip=True)
        datagen.fit(X)
                    
		y_pred = trained_model.predict_classes(X)
		y_prob = trained_model.predict(X)
		
		if type(metrics_array) is not list:
			metrics_array = [metrics_array]

		for metric in metrics_array:
			metric(X_file, X, Y, y_pred, y_prob)
			print('\n------------------------------------------\n')

		del X, Y
		print('\n==============================================\n')