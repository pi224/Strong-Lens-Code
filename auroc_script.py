import utils, models_preprocessing, metrics
from sklearn.model_selection import KFold, train_test_split
import numpy

model_function = models_preprocessing.compiledRegularizedConvnet
model = model_function()
auroc_Graph = metrics.aurocGraph
confusionMatrix = metrics.confusionMatrix
X_FILE = 'flipped.npy'
Y_FILE = 'flipped_labels.npy'

print('loading data ...')
data = numpy.load(X_FILE)
labels = numpy.load(Y_FILE)
print('done!')

Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, labels, test_size=0.3)

trained_model = utils.train(model, 26, Xtrain, Ytrain, None, None, True)

utils.test(trained_model, [confusionMatrix, auroc_Graph], Xtest, Ytest)
