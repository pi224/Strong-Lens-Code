import utils, models_preprocessing, metrics
import numpy
from keras.preprocessing.image import ImageDataGenerator

model_function = models_preprocessing.compiledRegularizedConvnet
auroc = metrics.auroc
accuracy = metrics.accuracy
text = metrics.basicTextMetrics
X_FILE = 'flipped.npy'
Y_FILE = 'flipped_labels.npy'

#print('loading data')
data = numpy.load(X_FILE)
labels = numpy.load(Y_FILE)
# data = numpy.load('../data/imadjust.npy')
# labels = numpy.load('../data/classification.npy')
# selection = numpy.random.choice(len(labels), len(labels)//2, replace=False)
# data = data[selection]
# labels = labels[selection]
#print('done')

# generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

# utils.epoch_curve(model_function, data, labels, 0.3, range(1, 41),
# 		[auroc, accuracy])

#run 5 fold cross validation on nn fed with image augmented data
utils.cross_validation(model_function, data, labels, 5, 30, [text])
