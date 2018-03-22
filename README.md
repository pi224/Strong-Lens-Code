# Strong-Lens-Code
This is a temporary home for code from the strong lenses project at MDST.

The data may be downloaded from the MBox folder dedicated to this project, per MDST policy. Everyone should be able to access that, as long as they login using their UMich IDs. Currently (17 September 2017) we are working with *imadjust.npy* with *classification.npy* as the labels.

# What should I do to get all the results I need?
It's very easy. First, make a directory outside of the repo called "results". This is where all data are going to be put. Then run
these files in any order (each one generates one part of the results)

cv_1.pbs  
cv_2.pbs  
cv_3.pbs  
cv_4.pbs  
cv_5.pbs  
cv_6.pbs  
cv_7.pbs  
cv_8.pbs  
cv_9.pbs  
cv_10.pbs  
large_1.pbs  
large_2.pbs

## WARNING:
These files are unmodified from the way they were on flux. Therefore, they still have some syntax specifically for working with
the University of Michigan's flux allocation. If you're having problems with lines like

```bash
if [-s "$PBS_NODEFILE" ...
```

or

```bash
module load python-anaoncda3
```
You can just remove those and nothing bad should happen

## What libraries did the experiments use?
keras verion 2.1.2  
tensorflow version 1.4.1  
numpy version 1.14.0  
scipy version 1.0.0  
python3 version 3.5.2


Below follows two sections:
* Code structure
* basic use cases/examples


### Overview of code structure
Our code is partitioned into 3 sections:
* Models
* Utilities
* Metrics

The models section contains the various networks and training settings 
we've tried, saved as functions written using the Keras python library.
For example, consider convet() in models_preprocessing.py:

```python
def convnet(input_shape=(101, 101, 4)):
	model = Sequential()
	model.add(Conv2D(64, (3, 3), strides=(2,2), activation='softplus',
					input_shape=input_shape))
	model.add(Conv2D(32, (3, 3), strides=(2,2), activation='softplus'))
	model.add(Conv2D(16, (3, 3), activation='softplus'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	model.add(Flatten())

	model.add(Dense(128, activation='softplus'))
	model.add(Dense(32, activation='softplus'))
	model.add(Dense(1, activation='sigmoid'))
	return model
```

This is how we record convnet's network architecture - as a function we call call
to obtain a keras neural network object.

However, neural networks are more than just architecture:

```python
def compiledConvnet(input_shape=(101, 101, 4)):
	model = convnet(input_shape)
	optimizer = Adam(lr = .0001, decay = 5e-5)
	model.compile(optimizer=optimizer,
	loss='binary_crossentropy',
	metrics=['accuracy'])

	return model
```

Keras not only creates the details of network architecture, it also allows us to
choose the hyperparameters for our network. So, to save favorable hyperparameter
selections, we create a function that calls convet() to get the network
architecture, and returns a Keras object that is "compiled" with the hyperparameters
and ready to train. All you need to do to use these functions is
```python
import models_preprocessing
```
in your terminal or in a script.


In utils.py, we have functions which take in objects from the models section above
and evaluate them in various ways. Details will be given below, but these functions
also take in data and other parameters not settable through compilation. Keep
in mind that we do not always give the object returned by the function from models
as a parameter. Sometimes we may give the function itself, so we can call it again
and again (remember that in python, functions are also objects).
For example, in utils.train we give the objects returned by the functions:
```python
utils.train(models_preprocessing.compiledConvnet(), num_epochs, X, Y)
```
but in utils.cross_validation we give the function:
```python
utils.cross_validation(models_preprocessing.compiledConvet, metrics ... ...)
```

What is metrics in utils.cross_validation? Conceptually, metrics are simply functions
whose output reflects model performance in some way. In our framework, we abstract
them away as well, to metrics.py. All metric function definitions follow this
structure:
```python
def <method_name>(testX, testY, yPred, yProb):
```
Where testX and testY are the feature vectors and labels of the test data,
yPred is actual predictions (for example, members of {0, 1} for binary classification)
and yProb is the model score - that is, the actual output of the model (for example,
members of [0, 1] in the real numbers for probabilitistic models).

We can use these metrics in utils methods, as objects. For example, say I want to use
auroc and accuracy in utils.test. Then I create an array and use it like this:
```python
metrics_array = [metrics.auroc, metrics.accuracy]

utils.test(model, metrics_array, X1, Y1, X2, Y2 ... )
```

That's the broad level overview. You can find details in the comments
in the code, and look at examples below:

### Example:

In this example, we will find give code for finding the optimal number of epochs
for a neural network called "convnet", drawing its learning curve, and then
confirming performance using cross validation.

Testing epochs 1-40 inclusive by using utils.epoch_curve:
```python
from models_preprocessing import compiledConvnet
from metrics import auroc, accuracy
import utils
import numpy

X = numpy.load('data')
Y = numpy.load('labels')

utils.epoch_curve(compiledConvnet, X, Y, validation_fraction=0.3,
		epochs_to_try=range(1, 41), evaluation_functions=[auroc, accuracy])
```
validation_fraction is the percentage of the data to use as the testing set.
Say the optimal number of epochs is 12, based on the figure from 
utils.epoch_curve. Then we may draw the learning curve for this number of epochs:
```python
from models_preprocessing import compiledConvnet
from metrics import auroc, accuracy
import utils
import numpy

X = numpy.load('data')
Y = numpy.load('labels')

utils.learning_curve(compiledConvnet, X, Y, fraction_test=0.3,
		num_iterations=20, num_epochs=12, evaluation_functions=[auroc, accuracy])
```
fraction_test serves the same function as validation_fraction in epoch_curve
Perhaps our learning curve looks good. We now run 5-fold cross validation to get
an idea for the stability of our model:
```python
from models_preprocessing import compiledConvnet
from metrics import auroc, accuracy
import utils
import numpy

X = numpy.load('data')
Y = numpy.load('labels')

utils.cross_validation(compiledConvnet, X, Y, numFolds=5, num_epochs=12,
				metrics_array=[auroc, accuracy])
```
