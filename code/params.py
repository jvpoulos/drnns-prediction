# Saves params to a dictionary
import cPickle as pickle
import numpy as np

# Define parameters for cross-validation
params_dict = {}
params_dict['batch_sizes'] = (23,46,161)
params_dict['dropouts'] = (0.25, 0.5)
params_dict['hidden_nodes'] = (32,64,128)
params_dict['activations'] = ('softmax', 'tanh', 'sigmoid')
params_dict['inits'] = ('glorot_uniform', 'glorot_normal', 'uniform')

# Save pickled version
pickle.dump(params_dict, open('params_dict.pkl', 'wb'))