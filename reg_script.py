import utils, models_preprocessing, metrics
import numpy

model_function = models_preprocessing.compiledRegularizedConvnet
auroc = metrics.auroc
text = metrics.basicTextMetrics

data = numpy.load('imadjust.npy')
labels = numpy.load('classification.npy')
utils.epoch_curve(model_function, data, labels, .3, range(1, 30), auroc)

#***NOTE: 30 is optimal number of epochs
#utils.cross_validation(model_function, text, data, labels, 5, 30, False)