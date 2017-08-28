from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import math
import numpy as np
import cPickle as pkl
import pandas as pd

from keras import backend as K
from keras.layers.core import *
from keras.models import *
from keras.layers import LSTM, Dense, Masking, Dropout, Activation, merge
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from keras import regularizers
from keras.optimizers import Adamax

from attention_utils import get_activations

SINGLE_ATTENTION_VECTOR = False # if True, the attention vector is shared across the input_dimensions where the attention is applied.
APPLY_ATTENTION_BEFORE_LSTM = True # if True, application of attention directly on features

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, nb_timesteps))(a)
    a = Dense(nb_timesteps, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


def model_attention_applied_after_lstm():
    inputs = Input(shape=(nb_timesteps, nb_features,))
    lstm_out = LSTM(nb_hidden, kernel_initializer=initialization, return_sequences=True)(inputs)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(output_dim, 
      activation=activation,
      kernel_regularizer=regularizers.l2(penalty),
      activity_regularizer=regularizers.l1(penalty))(attention_mul)
    model = Model(input=[inputs], output=output)
    return model

def model_attention_applied_before_lstm():
    inputs = Input(shape=(nb_timesteps, nb_features,))
    attention_mul = attention_3d_block(inputs)
    lstm_1 = LSTM(nb_hidden, kernel_initializer=initialization, return_sequences=True)(attention_mul)
    dropout_1 = Dropout(dropout)(lstm_1)
    lstm_2 = LSTM(nb_hidden, kernel_initializer=initialization, return_sequences=True)(dropout_1)
    dropout_2 = Dropout(dropout)(lstm_2)
    lstm_3 = LSTM(nb_hidden, kernel_initializer=initialization, return_sequences=False)(dropout_2)
    output = Dense(output_dim, 
      activation=activation,
      kernel_regularizer=regularizers.l2(penalty),
      activity_regularizer=regularizers.l1(penalty))(lstm_3)
    model = Model(input=[inputs], output=output)
    return model

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

X_val = pkl.load(open('data/{}_x_val_{}.np'.format(dataname,analysis), 'rb')) 
y_val = pkl.load(open('data/{}_y_val_{}.np'.format(dataname,analysis), 'rb')) 

# # Define network structure

epochs = int(sys.argv[-3])
nb_timesteps = 10
nb_features = X_train.shape[1]
output_dim = 1

# # Define model parameters

dropout = 0.7
penalty = 0.01
batch_size = 32
nb_hidden = 256
activation = 'linear'
initialization = 'glorot_normal'

# # Reshape X to three dimensions
# # Should have shape (batch_size, nb_timesteps, nb_features)

X_train = np.resize(X_train, (X_train.shape[0], nb_timesteps, X_train.shape[1]))

print('X_train shape:', X_train.shape)

X_val = np.resize(X_val, (X_val.shape[0], nb_timesteps, X_val.shape[1]))

print('X_val shape:', X_val.shape)

# Reshape y to two dimensions
# Should have shape (batch_size, output_dim)

y_train = np.resize(y_train, (y_train.shape[0], output_dim))

print('y_train shape:', y_train.shape)

y_val = np.resize(y_val, (y_val.shape[0], output_dim))

print('y_val shape:', y_val.shape)

# Initiate sequential model

if APPLY_ATTENTION_BEFORE_LSTM:
  m = model_attention_applied_before_lstm()
else:
  m = model_attention_applied_after_lstm()

# Configure learning process

m.compile(optimizer=Adamax(lr=0.002), 
  loss='mean_absolute_error', 
  metrics=['mean_absolute_error'])

print(m.summary())

# Prepare model checkpoints and callbacks

filepath="results/land-weights/weights-{val_mean_absolute_error:.2f}.hdf5"
checkpointer = ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=False)

TB = TensorBoard(log_dir='results/logs', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

# Training 
csv_logger = CSVLogger('results/training_log_{}.csv'.format(dataname), separator=',', append=True)

m.fit(X_train,
  y_train,
  verbose=1, 
  epochs=epochs, 
  batch_size=batch_size, 
  shuffle=True, 
  callbacks=[checkpointer,csv_logger,TB],
  validation_data=(X_val, y_val))

# Get attention weights on validation features
attention_vector = get_activations(m, X_val, print_shape_only=True, layer_name='attention_vec')[0][1]

attention_vector = np.mean(attention_vector, axis=1).squeeze() # mean across nb_timesteps

print('attention =', attention_vector)
print('attention shape =', attention_vector.shape)

np.savetxt('results/land-weights/attention.csv', attention_vector, delimiter=',') # save attentions to file

# plot

pd.DataFrame(attention_vector, columns=['attention (%)']).plot(kind='bar',title='Attention as function of features')
plt.savefig('results/land-weights/attention.png')