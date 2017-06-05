import pandas as pd
import numpy as np
from datetime import date
import nltk
import cPickle as pkl

# Read data
print('Reading data')

data = pd.read_csv("data/census-county/county-df.csv") # Gini and tenancy
del data['Unnamed: 0'] # rm row id

data_farmsize = pd.read_csv("data/census-county/county-df-farmsize.csv") # farmsize
del data_farmsize['Unnamed: 0'] # rm row id

# Split into training, validation, and test set
print('Split data')

num_training = (2480*3)  # train 1880-1900; test 1910-1950

num_training_farmsize = (2510*3) # train 1880-1900; test 1930-1950

X_train = data[data.columns[2:72]].values[:num_training]
X_test = data[data.columns[2:72]].values[num_training:]

X_train_farmsize = data_farmsize[data_farmsize.columns[2:72]].values[:num_training_farmsize]
X_test_farmsize = data_farmsize[data_farmsize.columns[2:72]].values[num_training_farmsize:]

y_train_gini = data["gini"].values[:num_training]
y_test_gini = data["gini"].values[num_training:]

y_train_tenancy = data["tenancy"].values[:num_training]
y_test_tenancy = data["tenancy"].values[num_training:]

y_train_farmsize = data_farmsize["farmsize"].values[:num_training_farmsize]
y_test_farmsize = data_farmsize["farmsize"].values[num_training_farmsize:]

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('X_train (farmsize) shape:', X_train_farmsize.shape)
print('X_test (farmsize) shape:', X_test_farmsize.shape)

# Save train and test sets to disk
print('Save to disk')

pkl.dump(X_train, open('data/X_train.np', 'wb'))
pkl.dump(X_test, open('data/X_test.np', 'wb'))

pkl.dump(X_train_farmsize, open('data/X_train_farmsize.np', 'wb'))
pkl.dump(X_test_farmsize, open('data/X_test_farmsize.np', 'wb'))

pkl.dump(y_train_gini, open('data/y_train_gini.np', 'wb'))
pkl.dump(y_test_gini, open('data/y_test_gini.np', 'wb'))

pkl.dump(y_train_tenancy, open('data/y_train_tenancy.np', 'wb'))
pkl.dump(y_test_tenancy, open('data/y_test_tenancy.np', 'wb'))

pkl.dump(y_train_farmsize, open('data/y_train_farmsize.np', 'wb'))
pkl.dump(y_test_farmsize, open('data/y_test_farmsize.np', 'wb'))