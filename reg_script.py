import utils, models_preprocessing, metrics
import numpy

model_function = models_preprocessing.compiledRegularizedConvnet
auroc = metrics.auroc
accuracy = metrics.accuracy
text = metrics.basicTextMetrics

data = numpy.load('imadjust.npy')
labels = numpy.load('classification.npy')

#utils.epoch_curve(model_function, data, labels, .3, range(1, 30), [auroc, accuracy])

utils.learning_curve(model_function, data, labels, fraction_test=0.3, num_iterations=20, num_epochs=2, evaluation_functions=[auroc,accuracy])


#***NOTE: 30 is optimal number of epochs
#utils.cross_validation(model_function, text, data, labels, 5, 30, False)