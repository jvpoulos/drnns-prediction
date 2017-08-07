## Trains best final model and saves weights at each epoch

from __future__ import print_function

import sys
import math
import numpy as np
import cPickle as pkl

from keras.models import Sequential
from keras.layers import LSTM, Dense, Masking, Dropout, Activation
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from keras import regularizers
from keras.optimizers import Adam

# Select gpu
import os
gpu = sys.argv[-4]
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]= "{}".format(gpu)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Load saved data

analysis = sys.argv[-1] # 'treated' or 'control'
dataname = sys.argv[-2]
print('Load saved {} data for analysis on {}'.format(dataname, analysis))

X_train = pkl.load(open('data/{}_x_train_{}.np'.format(dataname,analysis), 'rb')) 

y_train = pkl.load(open('data/{}_y_train_{}.np'.format(dataname,analysis), 'rb')) 

# Define network structure

epochs = int(sys.argv[-3])
nb_timesteps = 1
nb_features = X_train.shape[1]
output_dim = 1

# Define model parameters

dropout = 0.5
penalty = 0
batch_size = 128 
nb_hidden = 256
activation = 'sigmoid'
initialization = 'glorot_normal'

# # Reshape X to three dimensions
# # Should have shape (batch_size, nb_timesteps, nb_features)

X_train = np.resize(X_train, (X_train.shape[0], nb_timesteps, X_train.shape[1]))

print('X_train shape:', X_train.shape)

# Reshape y to two dimensions
# Should have shape (batch_size, output_dim)

y_train = np.resize(y_train, (y_train.shape[0], output_dim))

print('y_train shape:', y_train.shape)

# Initiate sequential model

print('Initializing model')

model = Sequential()
model.add(Masking(mask_value=0., input_shape=(nb_timesteps, nb_features))) # embedding for variable input lengths
model.add(LSTM(nb_hidden, return_sequences=True, kernel_initializer=initialization))
model.add(Dropout(dropout))  
model.add(LSTM(nb_hidden, return_sequences=True, kernel_initializer=initialization)) 
model.add(Dropout(dropout))
model.add(LSTM(nb_hidden, kernel_initializer=initialization))
model.add(Dropout(dropout))  
model.add(Dense(output_dim, 
  activation=activation,
  kernel_regularizer=regularizers.l2(penalty),
  activity_regularizer=regularizers.l1(penalty)))

# Configure learning process

model.compile(optimizer=Adam(lr=0.001),
              loss='mean_absolute_error',
              metrics=['mean_absolute_error'])

# Prepare model checkpoints and callbacks

filepath="results/land-weights/weights-{mean_absolute_error:.1f}.hdf5"
checkpointer = ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=False)

TB = TensorBoard(log_dir='results/logs', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

# Train model
print('Training')
csv_logger = CSVLogger('results/training_log_{}.csv'.format(dataname), separator=',', append=True)

model.fit(X_train,
  y_train,
  batch_size=batch_size,
  verbose=1,
  epochs=epochs,
  shuffle=True, 
  callbacks=[checkpointer,csv_logger,TB])