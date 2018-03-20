print('importing libraries')
import utils, models_preprocessing, metrics
import numpy as np
import pickle as pkl
import os
from sklearn.model_selection import train_test_split as split
from sys import argv

# ----------PARAMETERS-------------------
X_FILE = '../data/blurdec.npy'
Y_FILE = '../data/classification_filtered.npy'
SAVE_FILE = '../results/blurdec_results_indexes.pkl'
IMAGE_AUGMENTATION = False
REG = False

if len(argv) is not 6:
	print('improper number of parameters')
	exit(1)

X_FILE = argv[1]
Y_FILE = argv[2]
SAVE_FILE = argv[3]
IMAGE_AUGMENTATION = argv[4] == '1'
REG = argv[5] == '1'

print(X_FILE, Y_FILE, SAVE_FILE, IMAGE_AUGMENTATION, REG)
# ---------------------------------------

model_function = models_preprocessing.compiledConvnet
num_epochs = 1 #12
if REG:
	model = models_preprocessing.compiledRegularizedConvnet
	num_epochs = 1 #25
	print('using regularized convnet')
print('num epochs:', num_epochs)

text = metrics.basicTextMetrics
confMat = metrics.confusionMatrix
aurocGraph = metrics.aurocGraph
rec0 = metrics.rec0
rec10 = metrics.rec10
# check to make sure we're not saving over existing data
if os.path.isfile(SAVE_FILE):
	print('ERROR: SAVE_FILE already exists. \
			Will not override existing data')
	exit()

print('loading data')
data = np.load(X_FILE).astype('uint8')
labels = np.load(Y_FILE)

metrics_array = [
					text,
					confMat,
					aurocGraph,
					rec0,
					rec10
				]
results = utils.cross_validation(
									model_function,
									data,
									labels,
									10,
									num_epochs,
									metrics_array,
									IMAGE_AUGMENTATION,
									False
								)

with open(SAVE_FILE, 'wb') as file:
	pkl.dump(results, file)
