## Trains best final model and saves weights at each epoch

from __future__ import print_function
import sys
import math
import numpy as np
from itertools import product
import cPickle as pkl

from keras import backend as K
from keras.models import Sequential
from keras.layers import GRU, Dense, Masking, Dropout, Activation
from keras.callbacks import Callback,EarlyStopping, ModelCheckpoint,CSVLogger
from keras.optimizers import RMSprop

# import tensorflow as tf
# tf.python.control_flow_ops = tf

from utils import set_trace, plot_ROC

# Load saved data

print('Load saved data')

X_train = pkl.load(open('data/X_train.np', 'rb'))
X_val = pkl.load(open('data/X_val.np', 'rb'))

y_train = pkl.load(open('data/y_train_gini.np', 'rb'))
y_val = pkl.load(open('data/y_val_gini.np', 'rb'))

X_val = X_val[1:X_val.shape[0]] # drop first sample so batch size is divisible 
y_val = y_val[1:y_val.shape[0]]

# Define network structure

epochs = int(sys.argv[-1])
nb_timesteps = 1
nb_classes = 2
nb_features = X_train.shape[1]
output_dim = 1

# Define model parameters

batch_size = 13
dropout = 0.5
activation = 'sigmoid'
nb_hidden = 128
initialization = 'glorot_normal'

# # Reshape X to three dimensions
# # Should have shape (batch_size, nb_timesteps, nb_features)

#X_train = csr_matrix.toarray(X_train) # convert from sparse matrix to N dimensional array

X_train = np.resize(X_train, (X_train.shape[0], nb_timesteps, X_train.shape[1]))

print('X_train shape:', X_train.shape)

#X_val = csr_matrix.toarray(X_val) # convert from sparse matrix to N dimensional array

X_val = np.resize(X_val, (X_val.shape[0], nb_timesteps, X_val.shape[1]))

print('X_val shape:', X_val.shape)

# Reshape y to two dimensions
# Should have shape (batch_size, output_dim)

y_train = np.resize(y_train, (X_train.shape[0], output_dim))

print('y_train shape:', y_train.shape)

y_val = np.resize(y_val, (X_val.shape[0], output_dim))

print('y_val shape:', y_val.shape)

# Initiate sequential model

print('Initializing model')

model = Sequential()

# Stack layers
# expected input batch shape: (batch_size, nb_timesteps, nb_features)
# note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.
model.add(Masking(mask_value=0., batch_input_shape=(batch_size, nb_timesteps, nb_features))) # embedding for variable input lengths
model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization,
               batch_input_shape=(batch_size, nb_timesteps, nb_features)))
model.add(Dropout(dropout))  
model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization))  
model.add(Dropout(dropout))
model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization))  
model.add(Dropout(dropout)) 
model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization))  
model.add(Dropout(dropout)) 
model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization))  
model.add(Dropout(dropout)) 
model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization))  
model.add(Dropout(dropout))
model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization))  
model.add(Dropout(dropout))
model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization))  
model.add(Dropout(dropout))
model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization))  
model.add(Dropout(dropout))
model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization))  
model.add(Dropout(dropout))
model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization))  
model.add(Dropout(dropout))
model.add(GRU(nb_hidden, stateful=True, init=initialization))  
model.add(Dropout(dropout)) 
model.add(Dense(output_dim, activation=activation))

# Configure learning process

model.compile(optimizer='rmsprop',
              loss='mean_absolute_error',
              metrics=['mean_absolute_error'])

# Prepare model checkpoints and callbacks

filepath="results/weights/weights-{val_mean_absolute_error:.5f}.hdf5"
checkpointer = ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=False)

class LearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        print('\nLR: {:.6f}\n'.format(lr))

# Training 

print('Training')
for i in range(epochs):
    csv_logger = CSVLogger('results/training_log.csv', separator=',', append=True)
    print('Epoch', i+1, '/', epochs)
    model.fit(X_train,
              y_train,
              batch_size=batch_size,
              verbose=1,
              nb_epoch=1,
              shuffle=False, # turn off shuffle to ensure training data patterns remain sequential
              callbacks=[checkpointer,csv_logger,LearningRateTracker()], 
              validation_data=(X_val, y_val))
    model.reset_states()