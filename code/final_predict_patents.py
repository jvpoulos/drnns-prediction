## Loads best checkpointed model and makes prediciton on test set

from __future__ import print_function

import sys
import math
import numpy as np
import cPickle as pkl

from keras.models import Sequential
from keras.layers import LSTM, Dense, Masking, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard
from keras import regularizers
from keras.optimizers import Adadelta

# Load saved data

dataname = sys.argv[-1]
print('Load saved {} test data'.format(dataname))

X_train = pkl.load(open('data/{}_x_train.np'.format(dataname), 'rb')) 
X_test = pkl.load(open('data/{}_x_test.np'.format(dataname), 'rb')) 

y_train = pkl.load(open('data/{}_y_train.np'.format(dataname), 'rb')) 
y_test = pkl.load(open('data/{}_y_test.np'.format(dataname), 'rb')) 

# Define network structure

nb_timesteps = 1
nb_features = X_train.shape[1]
output_dim = 1

# Define model parameters

dropout = 0
penalty = 0 
batch_size = X_train.shape[0] # use entire
nb_hidden = 256
activation = 'linear'
initialization = 'glorot_normal'

# # Reshape X to three dimensions
# # Should have shape (batch_size, nb_timesteps, nb_features)

X_train = np.resize(X_train, (X_train.shape[0], nb_timesteps, X_train.shape[1]))
X_test= np.resize(X_test, (X_test.shape[0], nb_timesteps, X_test.shape[1]))

# Reshape y to two dimensions
# Should have shape (batch_size, output_dim)

y_train = np.resize(y_train, (y_train.shape[0], output_dim))
y_test = np.resize(y_test, (y_test.shape[0], output_dim))

# Initiate sequential model

print('Initializing model')

model = Sequential()
model.add(Masking(mask_value=0., input_shape=(nb_timesteps, nb_features))) # embedding for variable input lengths
model.add(LSTM(nb_hidden, return_sequences=True, kernel_initializer=initialization, dropout=dropout, recurrent_dropout=dropout))  
model.add(LSTM(nb_hidden, return_sequences=True, kernel_initializer=initialization, dropout=dropout, recurrent_dropout=dropout))  
model.add(LSTM(nb_hidden, kernel_initializer=initialization, dropout=dropout, recurrent_dropout=dropout))  
model.add(Dense(output_dim, 
  activation=activation,
  kernel_regularizer=regularizers.l2(penalty),
  activity_regularizer=regularizers.l1(penalty)))

# Load weights
filename = sys.argv[-2]
model.load_weights(filename)

# Configure learning process

Adadelta = Adadelta(clipnorm=5.) # Clip parameter gradients to a maximum norm of 5

model.compile(optimizer=Adadelta,
              loss='mean_absolute_error',
              metrics=['mean_absolute_error'])

print("Created model and loaded weights from file")

# Evaluation 

print('Generate predictions on test data')

y_pred_test = model.predict(X_test, batch_size=batch_size, verbose=1) # generate output predictions for test samples, batch-by-batch

np.savetxt("results/ok-pred/{}-test-{}.csv".format(dataname,filename), y_pred_test, delimiter=",")

# Get fits on training set for plots

print('Generate predictions on training set')

y_pred_train = model.predict(X_train, batch_size=batch_size, verbose=1) 

np.savetxt("results/ok-pred/{}-train-{}.csv".format(dataname,filename), y_pred_train, delimiter=",")