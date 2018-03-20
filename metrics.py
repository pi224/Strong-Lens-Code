from sklearn import metrics
import matplotlib.pyplot as plt
import numpy
#deprecated, don't use!
def basicTextMetrics(name, testX, testY, yPred, yProb):
	AUROC = metrics.roc_auc_score(testY, yProb)
	precision = metrics.precision_score(testY, yPred,
			average = 'binary')
	recall = metrics.recall_score(testY, yPred,
			average = 'binary')
	accuracy = metrics.accuracy_score(testY, yPred)
	
	print('\n' + name + ' AUROC:', AUROC)
	print('\n' + name + ' precision:', precision)
	print('\n' + name + ' recall:', recall)
	print('\n' + name + ' accuracy:', accuracy)
	return {
				'AUROC': AUROC,
				'precision': precision,
				'recall': recall,
				'accuracy': accuracy
			}

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

# todo: recall at 0
def rec0(name, testX, testY, yPred, yProb):
	#print('testY:', testY)
	yProb = yProb.flatten()
	#print('probs:', yProb)
	argrank = numpy.argsort(yProb)[::-1]
	rankedLabels = testY[argrank]
	#print('argrank', argrank)
	#print('ranked', rankedLabels)
	i = 0
	while rankedLabels[i] > 0:
		#print(rankedLabels[i])
		i+=1
		if i >= len(rankedLabels):
			break
	print('rec0', i)
	return i

# todo: recall at top 10
def rec10(name, testX, testY, yPred, yProb):
	yProb = yProb.flatten()
	argrank = numpy.argsort(yProb)[::-1]
	rankedLabels = testY[argrank]
	print(rankedLabels[0:20])
	i = 0
	errorCount = 0
	while errorCount < 10:
		if rankedLabels[i] < 1:
			errorCount+=1
		i+=1
		if i >= len(rankedLabels):
			break
	print('rec10', i)
	return i

#------------------------------------
#below this line are only metrics, no print

def auroc(testX, testY, yPred, yProb):
	return metrics.roc_auc_score(testY, yProb)

fast_accuracy = lambda pred, label: sum([1 if p == l else 0
					for p, l in zip(pred, label)]) / len(label)

def accuracy(testX, testY, yPred, yProb):
	return fast_accuracy(yPred, testY)
