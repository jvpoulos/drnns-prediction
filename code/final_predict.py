## Loads best checkpointed model and makes prediciton on test set

from __future__ import print_function
import numpy as np
from itertools import product
import cPickle as pkl
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

from keras.utils.visualize_util import plot, model_to_dot
from keras.models import Sequential
from keras.layers import GRU, Dense, Masking, Dropout, Activation
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

import tensorflow as tf
tf.python.control_flow_ops = tf

from utils import set_trace, plot_ROC

# Load saved data

print('Load saved test data')

X_test = pkl.load(open('data/X_test.np', 'rb'))

y_test = pkl.load(open('data/y_test.np', 'rb'))

# Label shift

lahead = 0 # number of days ahead that are used to make the prediction

if lahead!=0:
  y_test = np.roll(y_test,-lahead,axis=0)
else:
  pass

# Define network structure

nb_timesteps = 1 
nb_classes = 2
nb_features = 32015
output_dim = 1

# Define cross-validated model parameters

batch_size = 14
dropout = 0.25
activation = 'sigmoid'
nb_hidden = 128
initialization = 'glorot_normal'

# # Reshape X to three dimensions
# # Should have shape (batch_size, nb_timesteps, nb_features)

X_test = csr_matrix.toarray(X_test) # convert from sparse matrix to N dimensional array

X_test = np.resize(X_test, (X_test.shape[0], nb_timesteps, X_test.shape[1]))

print('X_test shape:', X_test.shape)

# Reshape y to two dimensions
# Should have shape (batch_size, output_dim)

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

# Visualize model

plot(model, to_file='results/final_model.png', # Plot graph of model
  show_shapes = True,
  show_layer_names = False)

model_to_dot(model,show_shapes=True,show_layer_names = False).write('results/final_model.dot', format='raw', prog='dot') # write to dot file

# Load weights
model.load_weights("results/weights/weights-00-0.5185.hdf5")

# Configure learning process

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Created model and loaded weights from file")

# Evaluation 
print('Evaluating results in terms of classification accuracy')

score = model.evaluate(X_test, y_test, batch_size=batch_size) # compute loss on test data, batch-by-batch
print("%s: %.2f%%" % (model.metrics_names[0], score[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

print('Evaluating results in terms of AUC')

y_probs = model.predict_proba(X_test, batch_size=batch_size, verbose=1)
print('AUC:' + str(roc_auc_score(y_test, y_probs)))

y_pred = model.predict(X_test, batch_size=batch_size, verbose=1) # generate output predictions for test samples, batch-by-batch

# Plot ROC curve
plot_ROC(y_test, y_pred)