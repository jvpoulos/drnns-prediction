# Train baseline 3 GRU layer-stacked "stateful" RNN for sequence classification
# https://keras.io/getting-started/sequential-model-guide/#examples


from sklearn.model_selection import TimeSeriesSplit, train_test_split
from keras.models import Sequential
from keras.layers import GRU, Dense, Masking, Dropout, Activation
from keras.callbacks import EarlyStopping
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

X_train = X_train[1:X_train.shape[0]] # drop first sample so batch sizes are even
y_train = y_train[1:y_train.shape[0]]

# Define cross-validation parameters
params_dict = pickle.load(open('params_dict.pkl', 'rb'))
batch_sizes = params_dict['batch_sizes']
dropouts = params_dict['dropouts']
activations = params_dict['activations']
hidden_nodes = params_dict['hidden_nodes']
inits = params_dict['inits'] 

validation_split = 0.2

#batch_size= 23 # must be multiple of sample size

#dropout = 0.5

# Define network structure
nb_epoch = 10 # max no. epochs
#nb_hidden = 32
nb_classes = 2
nb_features = X_train.shape[1]

nb_timesteps = 14 # use past two weeks for classification

output_dim = 2

# # Reshape X to three dimensions
# # Should have shape (batch_size, nb_timesteps, nb_features)

X_train = csr_matrix.toarray(X_train) # convert from sparse matrix to N dimensional array

X_train = np.resize(X_train, (X_train.shape[0], nb_timesteps, X_train.shape[1]))

print('X_train shape:', X_train.shape)

# Reshape y to two dimensions
# Should have shape (batch_size, output_dim)

y_train = np.resize(y_train, (X_train.shape[0], output_dim))

print('y_train shape:', y_train.shape)

# Matrix to store results
params_matrix = np.array([x for x in product(batch_sizes, dropouts, activations, hidden_nodes, inits)])
params_matrix = np.column_stack((params_matrix,
                                 np.zeros(params_matrix.shape[0]),
                                 np.zeros(params_matrix.shape[0]),
                                 np.zeros(params_matrix.shape[0]),
                                 np.zeros(params_matrix.shape[0]),
                                 np.zeros(params_matrix.shape[0])))

val_acc = []
val_loss = []
running_time = []

for param_idx in xrange(params_matrix.shape[0]):
    batch_size = int(params_matrix[param_idx, 0])
    dropout = int(params_matrix[param_idx, 1])
    activation = int(params_matrix[param_idx, 2])
    nb_hidden = int(params_matrix[param_idx, 3])
    inits = int(params_matrix[param_idx, 4])

  # Initiate sequential model

  model = Sequential()

  # Stack layers
  # expected input batch shape: (batch_size, nb_timesteps, nb_features)
  # note that we have to provide the full batch_input_shape since the network is stateful.
  # the sample of index i in batch k is the follow-up for the sample i in batch k-1.
  model.add(Masking(mask_value=0., batch_input_shape=(batch_size, nb_timesteps, nb_features))) # embedding for variable input lengths
  model.add(GRU(nb_hidden, return_sequences=True, stateful=True,
                 batch_input_shape=(batch_size, nb_timesteps, nb_features)))
  model.add(Dropout(dropout))  
  model.add(GRU(nb_hidden, return_sequences=True, stateful=True))  
  model.add(Dropout(dropout)) 
  model.add(GRU(nb_hidden, stateful=True))  
  model.add(Dropout(dropout)) 
  model.add(Dense(output_dim, activation=activation))

  # Configure learning process

  model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

  # Iterate on training data in batches

  early_stopping = EarlyStopping(monitor='val_loss', patience=1)

  model.fit(X_train, y_train,
            batch_size=batch_size, nb_epoch=nb_epoch,
            shuffle='batch', # shuffle in batch-sized chunks
            callbacks=[early_stopping], # stop early if validation loss not improving after 1 epoch
            validation_split = validation_split) # use last 20% of data for validation set

  # Evaluate best model performance on validation set

  score = model.evaluate(X_train[X_train.shape[0]-(X_train.shape[0]*validation_split):X_train.shape[0]], 
    y_train[y_train.shape[0]-(y_train.shape[0]*validation_split):y_train.shape[0]], 
    batch_size=batch_size)

  print 'Validation set {}: {} - {}: {}'.format(model.metrics_names[0],
    score[0],
      model.metrics_names[1],
      score[1])

# # Generate predictons on new data

# classes = model.predict_classes(X_test, batch_size=batch_size)
# proba = model.predict_proba(X_test, batch_size=batch_size)