from sklearn import metrics
import matplotlib.pyplot as plt
import numpy
#deprecated, don't use!
def basicTextMetrics(name, testX, testY, yPred, yProb):
	print('\n' + name + ' AUROC:', metrics.roc_auc_score(testY, yProb))
	print('\n' + name + ' precision:', metrics.precision_score(testY, yPred,
			average = 'binary'))
	print('\n' + name + ' recall:', metrics.recall_score(testY, yPred,
			average = 'binary'))
	print('\n' + name + ' accuracy:', metrics.accuracy_score(testY, yPred))
	return

#deprecated, don't use!
def aurocGraph(name, testX, testY, yPred, yProb):
	auroc_val = metrics.roc_auc_score(testY, yProb)
	print('auroc_val:', auroc_val)
	fpr, tpr, thresholds = metrics.roc_curve(testY, yProb, 1)
	return {'auroc': auroc_val, 'fpr': fpr, 'tpr': tpr}

#also deprecated
def confusionMatrix(name, testX, testY, yPred, yProb):
	matrix = numpy.asarray(metrics.confusion_matrix(testY, yPred,
				[1, 0]))
	print('\nConfusion Matrix:\n', 
			matrix, '\n')

	return matrix


#------------------------------------
#below this line are only metrics, no print

def auroc(testX, testY, yPred, yProb):
	return metrics.roc_auc_score(testY, yProb)

fast_accuracy = lambda pred, label: sum([1 if p == l else 0
					for p, l in zip(pred, label)]) / len(label)

def accuracy(testX, testY, yPred, yProb):
	return fast_accuracy(yPred, testY)