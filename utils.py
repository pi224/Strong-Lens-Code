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
		compiled_model.fit(trainX, trainY, epochs=epochs)
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
		y_pred = trained_model.predict_classes(X)
		y_prob = trained_model.predict(X)
		
		if type(metrics_array) is not list:
			metrics_array = [metrics_array]

		for metric in metrics_array:
			metric(X_file, X, Y, y_pred, y_prob)
			print('\n------------------------------------------\n')

		del X, Y
		print('\n==============================================\n')

def bootstrap(datum, num_samples):
	if type(datum) is str:
		datum = numpy.load(datum)

	selection = numpy.random.choice(datum.shape[0], num_samples, replace=True)
	sample = datum[selection]
	return sample