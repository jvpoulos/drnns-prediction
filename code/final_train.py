## Trains best final model and saves weights at each epoch

from __future__ import print_function
import numpy as np
from itertools import product
import cPickle as pkl
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

from keras.utils.visualize_util import plot, model_to_dot
from keras.models import Sequential
from keras.layers import GRU, Dense, Masking, Dropout, Activation
from keras.callbacks import Callback,EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger

import tensorflow as tf
tf.python.control_flow_ops = tf

from utils import set_trace, plot_ROC

# Load saved data

print('Load saved data')

X_train = pkl.load(open('data/X_train.np', 'rb'))
X_test = pkl.load(open('data/X_test.np', 'rb'))

y_train = pkl.load(open('data/y_train.np', 'rb'))
y_test = pkl.load(open('data/y_test.np', 'rb'))

X_train = X_train[1:X_train.shape[0]] # drop first sample so batch size is divisible 
y_train = y_train[1:y_train.shape[0]]

# Define network structure

epochs = 2
nb_timesteps = 1 
nb_classes = 2
nb_features = X_train.shape[1]
output_dim = 1

# Define cross-validated model parameters

batch_size = 14
dropout = 0.25
activation = 'sigmoid'
nb_hidden = 128
initialization = 'glorot_normal'

# # Reshape X to three dimensions
# # Should have shape (batch_size, nb_timesteps, nb_features)

X_train = csr_matrix.toarray(X_train) # convert from sparse matrix to N dimensional array

X_train = np.resize(X_train, (X_train.shape[0], nb_timesteps, X_train.shape[1]))

print('X_train shape:', X_train.shape)

X_test = csr_matrix.toarray(X_test) # convert from sparse matrix to N dimensional array

X_test = np.resize(X_test, (X_test.shape[0], nb_timesteps, X_test.shape[1]))

print('X_test shape:', X_test.shape)

# Reshape y to two dimensions
# Should have shape (batch_size, output_dim)

y_train = np.resize(y_train, (X_train.shape[0], output_dim))

print('y_train shape:', y_train.shape)

y_test = np.resize(y_test, (X_test.shape[0], output_dim))

print('y_test shape:', y_test.shape)

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
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Prepare callbacks
filepath="results/weights/weights-{val_acc:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=False)

tensorboard = TensorBoard(log_dir='results/logs', histogram_freq=0, write_graph=True, write_images=True)

# Training 

print('Training')
for i in range(epochs):
    print('Epoch', i+1, '/', epochs)
    csv_logger = CSVLogger('results/training-log.csv', separator=',', append=True)
    model.fit(X_train,
              y_train,
              batch_size=batch_size,
              verbose=1,
              nb_epoch=1,
              shuffle=False, # turn off shuffle to ensure training data patterns remain sequential
              callbacks=[checkpointer, csv_logger, tensorboard], 
              validation_data=(X_test, y_test))
    model.reset_states()