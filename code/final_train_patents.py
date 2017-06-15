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
from keras.optimizers import Adadelta

# Load saved data

print('Load saved data')

X_train = pkl.load(open('data/X_train_patents.np', 'rb'))
X_val = pkl.load(open('data/X_val_patents.np', 'rb'))

# y_train = pkl.load(open('data/y_train_sales.np', 'rb')) # sales
# y_val = pkl.load(open('data/y_val_sales.np', 'rb')) # sales

y_train = pkl.load(open('data/y_train_homesteads.np', 'rb')) # homesteads
y_val = pkl.load(open('data/y_val_homesteads.np', 'rb')) # homesteads

# Define network structure

epochs = int(sys.argv[-1])
nb_timesteps = 1
nb_features = X_train.shape[1]
output_dim = 1

# Define model parameters

#dropout = 0.5
dropout = 0.25
#penalty = 0.001 
penalty = 0
batch_size = 64
nb_hidden = 256
activation = 'linear'
initialization = 'glorot_normal'

# # Reshape X to three dimensions
# # Should have shape (batch_size, nb_timesteps, nb_features)

X_train = np.resize(X_train, (X_train.shape[0], nb_timesteps, X_train.shape[1]))

X_val= np.resize(X_val, (X_val.shape[0], nb_timesteps, X_val.shape[1]))

print('X_train shape:', X_train.shape)
print('X_val shape:', X_val.shape)

# Reshape y to two dimensions
# Should have shape (batch_size, output_dim)

y_train = np.resize(y_train, (y_train.shape[0], output_dim))

y_val = np.resize(y_val, (y_val.shape[0], output_dim))

print('y_train shape:', y_train.shape)
print('y_val shape:', y_train.shape)

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

# Configure learning process

Adadelta = Adadelta(clipnorm=5.) # Clip parameter gradients to a maximum norm of 5

model.compile(optimizer=Adadelta,
              loss='mean_absolute_error',
              metrics=['mean_absolute_error'])

# Prepare model checkpoints and callbacks

filepath="results/ok-weights/homesteads/weights-{val_mean_absolute_error:.2f}.hdf5"
checkpointer = ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=False)

#earlystop = EarlyStopping(monitor='val_mean_absolute_error', patience=5) # stops if val train error does not improve

TB = TensorBoard(log_dir='results/logs/patents', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

# Train model
print('Training')
csv_logger = CSVLogger('results/training_log_homesteads.csv', separator=',', append=True)

model.fit(X_train,
  y_train,
  batch_size=batch_size,
  verbose=1,
  epochs=epochs,
  shuffle=True,
  callbacks=[checkpointer,csv_logger,TB],
  validation_data=(X_val, y_val))