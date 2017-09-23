from sklearn import metrics
import matplotlib.pyplot as plt
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
	fpr, tpr, thresholds = metrics.roc_curve(testY, yProb)
	plt.title(name + ' ROC curve')
	plt.plot(tpr, fpr)
	plt.show()
	return


#------------------------------------
#below this line are only metrics, no print

def auroc(testX, testY, yPred, yProb):
	return metrics.roc_auc_score(testY, yProb)

fast_accuracy = lambda pred, label: sum([1 if p == l else 0
					for p, l in zip(pred, label)]) / len(label)

def accuracy(testX, testY, yPred, yProb):
	return fast_accuracy(yPred, testY)