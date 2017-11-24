import keras
import utils, metrics, models_preprocessing, numpy

convnet = models_preprocessing.convnet()
convnet.load_weights('../TrainedModels/convnet_imadjust_reg0.5')
#compile
optimizer = Adam(lr = .0001, decay = 5e-5)
convnet.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])