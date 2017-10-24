## Load best checkpointed model and make predictions

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
from keras.utils.vis_utils import plot_model, model_to_dot
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Masking, Dropout, Activation, Permute, Reshape, Input, Flatten, merge
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from keras import regularizers
from keras.optimizers import Adam

from attention_utils import get_activations

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
X_val = pkl.load(open('data/{}_x_val_{}.np'.format(dataname,analysis), 'rb'))
X_test = pkl.load(open('data/{}_x_test_{}.np'.format(dataname,analysis), 'rb')) 

y_train = pkl.load(open('data/{}_y_train_{}.np'.format(dataname,analysis), 'rb')) 
y_val = pkl.load(open('data/{}_y_val_{}.np'.format(dataname,analysis), 'rb')) 
y_test = pkl.load(open('data/{}_y_test_{}.np'.format(dataname,analysis), 'rb')) 

# Define network structure

nb_timesteps = 1
nb_features = X_train.shape[1]
output_dim = 1

# Define model parameters

dropout = 0.95
penalty = 0.55
batch_size = 32
nb_hidden = 256
activation = 'linear'
initialization = 'glorot_normal'

# # Reshape X to three dimensions
# # Should have shape (batch_size, nb_timesteps, nb_features)

X_train = np.resize(X_train, (X_train.shape[0], nb_timesteps, X_train.shape[1]))
X_val = np.resize(X_val, (X_val.shape[0], nb_timesteps, X_val.shape[1]))
X_test= np.resize(X_test, (X_test.shape[0], nb_timesteps, X_test.shape[1]))

# Reshape y to two dimensions
# Should have shape (batch_size, output_dim)

y_train = np.resize(y_train, (y_train.shape[0], output_dim))
y_val = np.resize(y_val, (y_val.shape[0], output_dim))
y_test = np.resize(y_test, (y_test.shape[0], output_dim))

# Initiate sequential model

print('Initializing model')

inputs = Input(shape=(nb_timesteps, nb_features,))
a = Permute((2, 1))(inputs)
a = Reshape((nb_features, nb_timesteps))(a)
a = Dense(nb_timesteps, activation='sigmoid')(a)
a_probs = Permute((2, 1), name='attention_vec')(a)
output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
lstm_1 = LSTM(nb_hidden, kernel_initializer=initialization, return_sequences=True)(output_attention_mul)
dropout_1 = Dropout(dropout)(lstm_1)
lstm_2 = LSTM(nb_hidden, kernel_initializer=initialization, return_sequences=False)(dropout_1)
output = Dense(output_dim, 
      activation=activation,
      kernel_regularizer=regularizers.l2(penalty),
      activity_regularizer=regularizers.l1(penalty))(lstm_2)
model = Model(input=[inputs], output=output)

print(model.summary())

# Visualize model

# plot_model(model, to_file='results/land/{}/model.png'.format(dataname), # Plot graph of model
#   show_shapes = False,
#   show_layer_names = False)

# model_to_dot(model,show_shapes=True,show_layer_names = False).write('results/land/{}/model.dot'.format(dataname), format='raw', prog='dot') # write to dot file

# Load weights
filename = sys.argv[-3]
model.load_weights(filename)

# Configure learning process

model.compile(optimizer=Adam(lr=0.001, clipnorm=5.), # Clip parameter gradients to a maximum norm of 5
              loss='mean_absolute_error',
              metrics=['mean_absolute_error'])

print("Created model and loaded weights from file")

# Evaluation 

print('Generate predictions on test data')

y_pred_test = model.predict(X_test, batch_size=batch_size, verbose=1) # generate output predictions for test samples, batch-by-batch

np.savetxt("{}-{}-test.csv".format(filename,dataname), y_pred_test, delimiter=",")

# Get fits on training and validation sets for plots

print('Generate predictions on training set')

y_pred_train = model.predict(X_train, batch_size=batch_size, verbose=1) 

np.savetxt("{}-{}-train.csv".format(filename,dataname), y_pred_train, delimiter=",")

print('Generate predictions on validation set')

y_pred_val = model.predict(X_val, batch_size=batch_size, verbose=1) 

np.savetxt("{}-{}-val.csv".format(filename,dataname), y_pred_val, delimiter=",")

# Get attention weights on validation features
attention_vector = get_activations(model, X_val, print_shape_only=True, layer_name='attention_vec')[0]

attention_vector = np.mean(attention_vector, axis=0).squeeze() # mean across # val samples

print('attention =', attention_vector)
print('attention shape =', attention_vector.shape)

np.savetxt('results/land/{}/attention.csv'.format(dataname), attention_vector, delimiter=',') # save attentions to file

# # plot

# pd.DataFrame(attention_vector, columns=['attention (%)']).plot(kind='bar',title='Attention as function of features')
# plt.savefig('results/land/{}/attention.png'.format(dataname))