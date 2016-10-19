# # Test set predictions using best baseline model (using all training data)

from sklearn.model_selection import TimeSeriesSplit, train_test_split
from keras.utils.visualize_util import plot
from keras.models import Sequential, load_model
from keras.layers import GRU, Dense, Masking, Dropout, Activation
from keras.callbacks import EarlyStopping
import numpy as np
from itertools import product
import cPickle as pkl
from scipy.sparse import csr_matrix
from utils import plot_ROC

import tensorflow as tf
tf.python.control_flow_ops = tf

# Load and preprocessed data

print('Loading and reading data')

X_train = pkl.load(open('data/X_train.np', 'rb'))
X_test = pkl.load(open('data/X_test.np', 'rb'))

y_train = pkl.load(open('data/y_train.np', 'rb'))
y_test = pkl.load(open('data/y_test.np', 'rb'))

X_train = X_train[1:X_train.shape[0]] # drop first sample so batch sizes are even
y_train = y_train[1:y_train.shape[0]]

# Define network structure

nb_epoch = 100 # max no. epochs
nb_classes = 2
nb_features = X_train.shape[1]

nb_timesteps = 14 # use past two weeks for classification

output_dim = 2

# Define cross-validated model parameters

batch_size = 46
dropout = 0.25
activation = 'sigmoid'
nb_hidden = 32
initialization = 'glorot_normal'

# # Reshape X to three dimensions
# # Should have shape (batch_size, nb_timesteps, nb_features)

X_train = csr_matrix.toarray(X_train) # convert from sparse matrix to N dimensional array

X_train = np.resize(X_train, (X_train.shape[0], nb_timesteps, X_train.shape[1]))

print('X_train shape:', X_train.shape)

X_test = csr_matrix.toarray(X_test) # convert from sparse matrix to N dimensional array

X_test = np.resize(X_test, (X_test.shape[0], nb_timesteps, X_test.shape[1]))

print('X_test shape:', X_train.shape)

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
model.add(GRU(nb_hidden, stateful=True, init=initialization))  
model.add(Dropout(dropout)) 
model.add(Dense(output_dim, activation=activation))

# Configure learning process

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Iterate on training data in batches

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

model.fit(X_train, y_train,
          batch_size=batch_size, nb_epoch=nb_epoch,
          shuffle='batch', # shuffle in batch-sized chunks
          callbacks=[early_stopping]) # stop early if validation loss not improving after 2 epochs

model.save('best_baseline_model.h5')  # creates a HDF5 file 
# model = load_model('best_baseline_model.h5')

# Evaluate test set performance

print('Predicting on test data')
y_score = model.predict(X_test)

print('Evaluating results')

scores = model.evaluate(X_test, y_test, batch_size=batch_size)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

plot_ROC(y_test[:, 0], y_score[:, 0])

# Save graph of model
plot(model, to_file='best_baseline_model.png')