import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import KFold, train_test_split

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

def train(compiled_model, epochs, steps_per_epoch=None, generator,
				valid_generator=None, print_summary=False):
	model = compiled_model
	if print_summary:
		print(model.summary())
	if steps_per_epoch is None:
		steps_per_epoch = len(generator) // 32
	
	try:
		if valid_generator is None:
			model.fit_generator(generator, steps_per_epoch=steps_per_epoch,
								epochs=epochs)
		else:
			model.fit_generator(generator, steps_per_epoch=steps_per_epoch,
						epochs=epochs, validation_data=valid_generator,
						validation_steps=steps_per_epoch)
	except KeyboardInterrupt:
		print('KeyboardInterrupt - returning current model')
		return model

	return model


def test(trained_model, metrics_array, *argv):
	testing_sets = [(0, 0)]

	try:
		generators = argv
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

#TODO: DEBUG!!!!
def learning_curve(model_function, data, labels, fraction_test,
							num_iterations, num_epochs, evaluation_functions):
	if evaluation_functions is not list:
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