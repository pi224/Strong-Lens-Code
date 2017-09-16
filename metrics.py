from sklearn import metrics
import matplotlib.pyplot as plt

def basicTextMetrics(name, testX, testY, yPred, yProb):
	print('\n' + name + ' AUROC:', metrics.roc_auc_score(testY, yProb))
	print('\n' + name + ' precision:', metrics.precision_score(testY, yPred,
			average = 'binary'))
	print('\n' + name + ' recall:', metrics.recall_score(testY, yPred,
			average = 'binary'))
	print('\n' + name + ' accuracy:', metrics.accuracy_score(testY, yPred))
	return

def aurocGraph(name, testX, testY, yPred, yProb):
	fpr, tpr, thresholds = metrics.roc_curve(testY, yProb)
	plt.title(name + ' ROC curve')
	plt.plot(tpr, fpr)
	plt.show()
	return