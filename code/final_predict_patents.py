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

print('Load saved test data')

X_train = pkl.load(open('data/X_train_patents.np', 'rb'))
X_val = pkl.load(open('data/X_val_patents.np', 'rb'))
X_test = pkl.load(open('data/X_test_patents.np', 'rb'))

y_train = pkl.load(open('data/y_train_sales.np', 'rb')) # sales
y_val = pkl.load(open('data/y_val_sales.np', 'rb')) 
y_test = pkl.load(open('data/y_test_sales.np', 'rb')) 

# Define network structure

nb_timesteps = 1
nb_features = X_train.shape[1]
output_dim = 1

# Define model parameters

dropout = 0
penalty = 0 
batch_size = 64
nb_hidden = 256
activation = 'linear'
initialization = 'glorot_normal'

# # Reshape X to three dimensions
# # Should have shape (batch_size, nb_timesteps, nb_features)

X_train = np.resize(X_train, (X_train.shape[0], nb_timesteps, X_train.shape[1]))
X_val= np.resize(X_val, (X_val.shape[0], nb_timesteps, X_val.shape[1]))
X_test= np.resize(X_test, (X_test.shape[0], nb_timesteps, X_test.shape[1]))

# Reshape y to two dimensions
# Should have shape (batch_size, output_dim)

y_train = np.resize(y_train, (y_train.shape[0], output_dim))
y_val = np.resize(y_val, (y_val.shape[0], output_dim))
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
model.load_weights(sys.argv[-1])

# Configure learning process

Adadelta = Adadelta(clipnorm=5.) # Clip parameter gradients to a maximum norm of 5

model.compile(optimizer=Adadelta,
              loss='mean_absolute_error',
              metrics=['mean_absolute_error'])

print("Created model and loaded weights from file")

# Evaluation 

print('Generate predictions on test data')

y_pred_test = model.predict(X_test, batch_size=batch_size, verbose=1) # generate output predictions for test samples, batch-by-batch

np.savetxt("results/ok-pred/sales-test-pred.csv", y_pred_test, delimiter=",")

# Get fits on training and validation set for plots

print('Generate predictions on training set')

y_pred_train = model.predict(X_train, batch_size=batch_size, verbose=1) 

np.savetxt("results/ok-pred/sales-train-pred.csv", y_pred_train, delimiter=",")

print('Generate predictions on validation set')

y_pred_val = model.predict(X_val, batch_size=batch_size, verbose=1) 

np.savetxt("results/ok-pred/sales-val-pred.csv", y_pred_val, delimiter=",")