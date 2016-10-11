# Train baseline 3 GRU layer-stacked RNN for sequence classification
# https://keras.io/getting-started/sequential-model-guide/#examples


from sklearn.model_selection import TimeSeriesSplit, train_test_split
from keras.models import Sequential
from keras.layers import GRU, Dense, Masking, Dropout, Activation
import numpy as np
import cPickle as pkl
from scipy.sparse import csr_matrix

import tensorflow as tf
tf.python.control_flow_ops = tf

def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

# Load and preprocessed data

X_train = pkl.load(open('data/X_train.np', 'rb'))
X_test = pkl.load(open('data/X_test.np', 'rb'))

y_train = pkl.load(open('data/y_train.np', 'rb'))
y_test = pkl.load(open('data/y_test.np', 'rb'))

 
# Time series cross-validation 
tscv = TimeSeriesSplit(n_splits=3)

for train_index, test_index in tscv.split(X_train):
	X_train_cv, X_val = X_train[train_index], X_train[test_index]  
	y_train_cv, y_val = y_train[train_index], y_train[test_index]

# Define data and learning parameters
batch_size=64
nb_epoch = 5
nb_hidden = 32

nb_timesteps = X_train_cv.shape[0] # use all inputs/timesteps for classification
nb_features = X_train_cv.shape[1]

output_dim = nb_timesteps

# Reshape X's to three dimensions
# Should have shape (batch_size, nb_timesteps, nb_features)

X_train_cv = csr_matrix.toarray(X_train_cv) # convert from sparse matrix to N dimensional array
X_train_cv = np.expand_dims(X_train_cv, axis=0)

X_val = csr_matrix.toarray(X_val)

X_val = np.pad(X_val, ((nb_timesteps-X_val.shape[0],0), (0,0)), mode='constant', constant_values=0) # pad with zeros

X_val = np.expand_dims(X_val, axis=0)

print('X_train_cv shape:', X_train_cv.shape)
print('X_val shape:', X_val.shape)

# Reshape y's to two dimensions
# Should have shape (batch_size, nb_classes)

y_train_cv = np.expand_dims(y_train_cv, axis=0)

y_val = np.pad(y_val, ((nb_timesteps-y_val.shape[0],0)), mode='constant', constant_values=0) # pad with zeros
y_val = np.expand_dims(y_val, axis=0)

print('y_train_cv shape:', y_train_cv.shape)
print('y_val shape:', y_val.shape)

# Initiate sequential model
# expected input data shape: (batch_size, timesteps, nb_features)

model = Sequential()

# Stack layers
model.add(Masking(mask_value=0., input_shape=(nb_timesteps, nb_features))) # embedding for variable input lengths
model.add(GRU(nb_hidden, return_sequences=True,
               input_shape=(nb_timesteps, nb_features)))  
model.add(GRU(nb_hidden, return_sequences=True))  
model.add(GRU(nb_hidden))  
model.add(Dense(output_dim, activation='softmax'))

# Configure learning process

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Iterate on training data in batches

model.fit(X_train_cv, y_train_cv,
          batch_size=batch_size, nb_epoch=nb_epoch,
          validation_data=(X_val, y_val))

# Evaluate performance on validation set

score = model.evaluate(X_val, y_val, batch_size=batch_size)

print 'Validation set {}: {} - {}: {}'.format(model.metrics_names[0],
	score[0],
    model.metrics_names[1],
    score[1])


# # Generate predictons on new data

# classes = model.predict_classes(X_test, batch_size=batch_size)
# proba = model.predict_proba(X_test, batch_size=batch_size)