import numpy
from sklearn.model_selection import KFold

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)

def train(compiled_model, epochs, trainX_file, trainY_file,
				validX_file = None, validY_file = None, print_summary=True):
	model = compiled_model
	if print_summary:
		print(model.summary())
	validX = None
	validY = None
	if type(validX_file) is str:
		validX = numpy.load(validX_file)
		validY = numpy.load(validY_file)
	else:
		trainX = trainX_file
		trainY = trainY_file

	if type(trainX_file) is str:
		trainX = numpy.load(trainX_file)
		trainY = numpy.load(trainY_file)
	else:
		trainX = trainX_file
		trainY = trainY_file
	try:
		if validX is None:
			model.fit(trainX, trainY, epochs=epochs)
		else:
			model.fit(trainX, trainY, epochs=epochs,
						validation_data=(validX, validY))
	except KeyboardInterrupt:
		print('KeyboardInterrupt - returning current model')
		return model

	return model


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
		y_pred = trained_model.predict_classes(X)
		y_prob = trained_model.predict(X)
		
		if type(metrics_array) is not list:
			metrics_array = [metrics_array]

		for metric in metrics_array:
			metric(X_file, X, Y, y_pred, y_prob)
			print('\n------------------------------------------\n')

		del X, Y
		print('\n==============================================\n')

def cross_validation(model_function, metrics_array,
			validX, validY, numFolds, num_epochs, print_summary = False):
	#load the data first
	if type(validX) is str:
		validX = numpy.load(validX)
		validY = numpy.load(validY)

	#partition into folds
	kf = KFold(n_splits=numFolds)

	counter = 0
	for train_index, test_index in kf.split(validX):
		print('Fold ', counter, '\n========================================\n')

		trainX, trainY = validX[train_index], validY[train_index]
		testX, testY = validX[test_index], validY[test_index]

		#train the model, and print out summary if on the first epoch
		print_summary = False
		if counter is 0 and print_summary:
			print_summary = True
		trained_model = train(model_function(), num_epochs,
					trainX, trainY, testX, testY, print_summary)

		y_pred = trained_model.predict_classes(testX)
		y_prob = trained_model.predict(testX)

		if type(metrics_array) is not list:
			metrics_array = [metrics_array]

		for metric in metrics_array:
			metric('Fold '+str(counter), testX, testY, y_pred, y_prob)
			print('------------------------------------------\n')

		counter += 1

def bootstrap(datum, num_samples):
	if type(datum) is str:
		datum = numpy.load(datum)

	selection = numpy.random.choice(datum.shape[0], num_samples, replace=True)
	sample = datum[selection]
	return sample