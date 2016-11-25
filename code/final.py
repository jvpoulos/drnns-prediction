# # Test set predictions using best final model (using all training data)

from sklearn.model_selection import TimeSeriesSplit, train_test_split
from keras.utils.visualize_util import plot
from keras.models import Sequential
from keras.layers import GRU, Dense, Masking, Dropout, Activation
from keras.callbacks import EarlyStopping
import numpy as np
from itertools import product
import cPickle as pkl
from scipy.sparse import csr_matrix
from utils import set_trace, plot_ROC
from sklearn.metrics import roc_curve, auc, roc_auc_score

import tensorflow as tf
tf.python.control_flow_ops = tf

# Load saved data

print('Load saved data')

X_train = pkl.load(open('data/X_train.np', 'rb'))
X_test = pkl.load(open('data/X_test.np', 'rb'))

y_train = pkl.load(open('data/y_train.np', 'rb'))
y_test = pkl.load(open('data/y_test.np', 'rb'))

X_train = X_train[1:X_train.shape[0]] # drop first sample so batch size is divisible 
y_train = y_train[1:y_train.shape[0]]

# Label shift

lahead = 0 # number of days ahead that are used to make the prediction

if lahead!=0:
  y_train = np.roll(y_train,-lahead,axis=0)
  y_test = np.roll(y_test,-lahead,axis=0)
else:
  pass

# Define network structure

epochs = 25 
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
              metrics=['binary_accuracy'])

plot(model, to_file='results/final_model.png', # Plot graph of model
  show_shapes = True,
  show_layer_names = False)

# Training 

early_stopping = EarlyStopping(monitor='binary_accuracy', patience=1)

print('Training')
for i in range(epochs):
    print('Epoch', i+1, '/', epochs)
    model.fit(X_train,
              y_train,
              batch_size=batch_size,
              verbose=1,
              nb_epoch=1,
              shuffle=False, # turn off shuffle to ensure training data patterns remain sequential
              callbacks=[early_stopping])  # stop early if training loss not improving after 1 epoch
    model.reset_states()

# Evaluation 
print('Evaluating results in terms of classification accuracy')

loss = model.evaluate(X_test, y_test, batch_size=batch_size) # compute loss on test data, batch-by-batch
print("%s: %.2f%%" % (model.metrics_names[1], loss[1]*100))

print('Evaluating results in terms of AUC')

y_probs = model.predict_proba(X_test, batch_size=batch_size, verbose=1)
print('AUC ' + str(roc_auc_score(y_test, y_probs)))

y_pred = model.predict(X_test, batch_size=batch_size, verbose=1) # generate output predictions for test samples, batch-by-batch

# Plot ROC curve
plot_ROC(y_test, y_pred)