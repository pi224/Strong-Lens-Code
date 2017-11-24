import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import KFold, train_test_split
from metrics import aurocGraph, confusionMatrix

def bootstrap(datum, num_samples, random_seed = None):
	if random_seed is not None:
		numpy.random.seed(random_seed)

	if type(datum) is str:
		datum = numpy.load(datum)

	selection = numpy.random.choice(datum.shape[0], num_samples, replace=True)
	sample = datum[selection]
	return sample

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)

def train(compiled_model, epochs, trainX_file, trainY_file,
				validX_file = None, validY_file = None, print_summary=False):
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

#deprecated, don't use!
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
	results = []
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
		dataset_result = []
		for metric in metrics_array:
			values = metric(X_file, X, Y, y_pred, y_prob)
			dataset_result.append(values)
			print('\n------------------------------------------\n')

		del X, Y
		print('\n==============================================\n')
		results.append(dataset_result)
	return results

def cross_validation(model_function, validX, validY, numFolds, num_epochs, 
					metrics_array, print_summary = False):
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


def learning_curve(model_function, data, labels, fraction_test,
							num_iterations, num_epochs, evaluation_functions):
	if type(evaluation_functions) is not list:
		evaluation_functions = [evaluation_functions]

	Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, labels,
											test_size=fraction_test)

	iteration_size = Ytrain.shape[0] // num_iterations

	data_boundary  = iteration_size
	test_results = [[]] * len(evaluation_functions)
	train_results = [[]] * len(evaluation_functions)
	num_data_points = []
	for i in range(num_iterations):
		currentX = Xtrain[0:data_boundary]
		currentY = Ytrain[0:data_boundary]
		trained_model = train(model_function(), num_epochs, currentX, currentY)

		#evaluate on train data
		y_pred = trained_model.predict_classes(currentX)
		y_prob = trained_model.predict(currentX)
		train_results = [r + [f(currentX, currentY, y_pred, y_prob)]
							for r, f in zip(train_results, evaluation_functions)]

		#evaluate on test data
		y_pred = trained_model.predict_classes(Xtest)
		y_prob = trained_model.predict(Xtest)
		test_results = [r + [f(Xtest, Ytest, y_pred, y_prob)]
							for r, f in zip(test_results, evaluation_functions)]
		
		num_data_points.append(data_boundary)
		data_boundary += iteration_size

	for train_result, test_result in zip(train_results, test_results):
		#lines
		plt.plot(num_data_points, test_result, color='r')
		plt.plot(num_data_points, train_result, color='g')
		
		#legend
		red_patch = mpatches.Patch(color='red', label='testing performance')
		green_patch = mpatches.Patch(color='green', label='training performance')
		plt.legend(handles=[red_patch, green_patch])
		
		plt.show()

	return {
				'train_size': num_data_points, #aka x-axis
				'train_plot': train_results, #aka train curve
				'test_plot':test_results #aka test curve
			}

def epoch_curve(model_function, data, labels, validation_fraction,
							epochs_to_try, evaluation_functions):
	Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, labels,
											test_size=validation_fraction)
	if type(evaluation_functions) is not list:
		evaluation_functions = [evaluation_functions]

	model = train(model_function(), epochs_to_try[0], Xtrain, Ytrain)

	test_results = [[]] * len(evaluation_functions)
	epochs = []
	for i in range(1, len(epochs_to_try)):
		y_pred = model.predict_classes(Xtest)
		y_prob = model.predict(Xtest)
		test_results = [r + [f(Xtest, Ytest, y_pred, y_prob)]
							for r, f in zip(test_results, evaluation_functions)]
		epochs.append(epochs_to_try[i-1])

		num_epochs = epochs_to_try[i] - epochs_to_try[i-1]
		model = train(model, num_epochs, Xtrain, Ytrain)

	y_pred = model.predict_classes(Xtest)
	y_prob = model.predict(Xtest)
	test_results = [r + [f(Xtest, Ytest, y_pred, y_prob)]
							for r, f in zip(test_results, evaluation_functions)]
	epochs.append(epochs_to_try[-1])

	print(test_results)
	print(epochs)

	for test_result in test_results:
		plt.plot(epochs, test_result, color='blue')
		plt.show()

	#test_results is y axis, epochs is x axis
	return {'epochs': epochs, 'results': test_results}


fast_accuracy = lambda pred, label: sum([1 if p == l else 0
					for p, l in zip(pred, label)]) / len(label)
def probabilistic_to_binary(probabilities, labels):
	if len(probabilities[0].shape) > 1:
		probabilities = numpy.asarray([p[0] for p in probabilities])

	binary = numpy.asarray([0 if p <0.5 else 1 for p in probabilities])
	flipped = numpy.asarray([1 if b is 0 else 0 for b in binary])
	if fast_accuracy(binary, labels) >= fast_accuracy(flipped, labels):
		return binary
	return flipped


def cross_validation_generator(model_function, validX, validY,
		generator, numFolds, num_epochs, batch_size,
		metrics_array, print_summary = False):
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

		trained_model = model_function()
		

		# trained_model = train(model_function(), num_epochs, currentX, currentY)
		generator.fit(trainX)
		trained_model.fit_generator(generator.flow(trainX, trainY),
					len(trainX)//batch_size, epochs=num_epochs)

		# y_pred = trained_model.predict_classes(currentX)
		# y_prob = trained_model.predict(currentX)
		y_prob = trained_model.predict_generator(generator.flow(testX, testY,),
					len(testY)//batch_size)
		y_pred = probabilistic_to_binary(y_prob, testY)

		if type(metrics_array) is not list:
			metrics_array = [metrics_array]

		for metric in metrics_array:
			metric('Fold '+str(counter), testX, testY, y_pred, y_prob)
			print('------------------------------------------\n')

		counter += 1



def learning_curve_generator(model_function, data, labels, fraction_test,
		generator, batch_size, num_iterations, num_epochs, evaluation_functions):
	if type(evaluation_functions) is not list:
		evaluation_functions = [evaluation_functions]

	Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, labels,
											test_size=fraction_test)
	Xtest = Xtest[0: (len(Ytest)//batch_size)*batch_size]
	Ytest = Ytest[0: (len(Ytest)//batch_size)*batch_size]

	iteration_size = Ytrain.shape[0] // num_iterations

	data_boundary  = iteration_size
	test_results = [[]] * len(evaluation_functions)
	train_results = [[]] * len(evaluation_functions)
	num_data_points = []
	for i in range(num_iterations):
		currentX = Xtrain[0:data_boundary]
		currentY = Ytrain[0:data_boundary]
		currentX = currentX[0: (len(currentY)//batch_size)*batch_size]
		currentY = currentY[0: (len(currentY)//batch_size)*batch_size]
		
		# trained_model = train(model_function(), num_epochs, currentX, currentY)
		generator.fit(currentX)
		trained_model = model_function()
		trained_model.fit_generator(generator.flow(currentX, currentY),
					len(currentX)//batch_size, epochs = num_epochs)

		#evaluate on train data
		# y_pred = trained_model.predict_classes(currentX)
		# y_prob = trained_model.predict(currentX)
		y_prob = trained_model.predict_generator(
				generator.flow(currentX, currentY,), len(currentY)//batch_size)
		y_pred = probabilistic_to_binary(y_prob, currentY)
		train_results = [r + [f(currentX, currentY, y_pred, y_prob)]
							for r, f in zip(train_results, evaluation_functions)]

		#evaluate on test data
		# y_pred = trained_model.predict_classes(Xtest)
		# y_prob = trained_model.predict(Xtest)
		y_prob = trained_model.predict_generator(generator.flow(Xtest, Ytest,),
					len(Ytest)//batch_size)
		y_pred = probabilistic_to_binary(y_prob, Ytest)
		test_results = [r + [f(Xtest, Ytest, y_pred, y_prob)]
							for r, f in zip(test_results, evaluation_functions)]
		
		num_data_points.append(data_boundary)
		data_boundary += iteration_size

	for train_result, test_result in zip(train_results, test_results):
		#lines
		plt.plot(num_data_points, test_result, color='r')
		plt.plot(num_data_points, train_result, color='g')
		
		#legend
		red_patch = mpatches.Patch(color='red', label='testing performance')
		green_patch = mpatches.Patch(color='green', label='training performance')
		plt.legend(handles=[red_patch, green_patch])
		
		plt.show()


def epoch_curve_generator(model_function, data, labels,
		generator, batch_size, validation_fraction, epochs_to_try,
		evaluation_functions):
	Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, labels,
											test_size=validation_fraction)
	Xtest = Xtest[0: (len(Ytest)//batch_size)*batch_size]
	Ytest = Ytest[0: (len(Ytest)//batch_size)*batch_size]
	print(len(Xtrain))


	if type(evaluation_functions) is not list:
		evaluation_functions = [evaluation_functions]

	generator.fit(Xtrain)
	trained_model = model_function()
	# model = train(model_function(), epochs_to_try[0], Xtrain, Ytrain)
	trained_model.fit_generator(generator.flow(Xtrain, Ytrain),
				len(Xtrain)//batch_size, epochs = epochs_to_try[0],
				max_queue_size = 2)

	test_results = [[]] * len(evaluation_functions)
	epochs = []
	for i in range(1, len(epochs_to_try)):
		# y_pred = model.predict_classes(Xtest)
		# y_prob = model.predict(Xtest)
		y_prob = trained_model.predict_generator(generator.flow(Xtest, Ytest,),
					len(Ytest)//batch_size)
		y_pred = probabilistic_to_binary(y_prob, Ytest)

		test_results = [r + [f(Xtest, Ytest, y_pred, y_prob)]
							for r, f in zip(test_results, evaluation_functions)]
		epochs.append(epochs_to_try[i-1])

		num_epochs = epochs_to_try[i] - epochs_to_try[i-1]
		# model = train(model, num_epochs, Xtrain, Ytrain)
		del y_prob, y_pred
		trained_model.fit_generator(generator.flow(Xtrain, Ytrain),
				len(Ytrain)//batch_size, epochs = num_epochs,
				max_queue_size = 2)

	y_prob = trained_model.predict_generator(generator.flow(Xtest, Ytest,),
					len(Ytest)//batch_size)
	y_pred = probabilistic_to_binary(y_prob, Ytest)
	test_results = [r + [f(Xtest, Ytest, y_pred, y_prob)]
							for r, f in zip(test_results, evaluation_functions)]
	epochs.append(epochs_to_try[-1])
	del y_prob, y_pred

	print(test_results)
	print(epochs)

	for test_result in test_results:
		plt.plot(epochs, test_result, color='blue')
		plt.show()