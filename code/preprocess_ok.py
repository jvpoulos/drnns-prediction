import pandas as pd
import numpy as np
import cPickle as pkl

# Read data
print('Reading data')

county_x_train = pd.read_csv("data/census-county/county-x-train.csv") # Gini and tenancy
county_x_val = pd.read_csv("data/census-county/county-x-val.csv") 
county_x_test = pd.read_csv("data/census-county/county-x-test.csv") 

county_x2_train = pd.read_csv("data/census-county/county-x2-train.csv") # farmsize
county_x2_val = pd.read_csv("data/census-county/county-x2-val.csv") 
county_x2_test = pd.read_csv("data/census-county/county-x2-test.csv") 

X_train = county_x_train[county_x_train.columns[3:72]]
X_val = county_x_val[county_x_val.columns[3:72]]
X_test = county_x_test[county_x_test.columns[3:72]]

X_train_farmsize = county_x2_train[county_x2_train.columns[3:72]]
X_val_farmsize = county_x2_val[county_x2_val.columns[3:72]]
X_test_farmsize = county_x2_test[county_x2_test.columns[3:72]]

y_train_gini = county_x_train["gini"]
y_val_gini = county_x_val["gini"]
y_test_gini = county_x_test["gini"]

y_train_tenancy = county_x_train["tenancy"]
y_val_tenancy = county_x_val["tenancy"]
y_test_tenancy = county_x_test["tenancy"]

y_train_farmsize = county_x2_train["farmsize"]
y_val_farmsize= county_x2_val["farmsize"]
y_test_farmsize = county_x2_test["farmsize"]

print('X_train shape:', X_train.shape)
print('X_val shape:', X_val.shape)
print('X_test shape:', X_test.shape)

print('X_train (farmsize) shape:', X_train_farmsize.shape)
print('X_val (farmsize) shape:', X_val_farmsize.shape)
print('X_test (farmsize) shape:', X_test_farmsize.shape)

# Save train and test sets to disk
print('Save to disk')

pkl.dump(X_train, open('data/X_train.np', 'wb'))
pkl.dump(X_val, open('data/X_val.np', 'wb'))
pkl.dump(X_test, open('data/X_test.np', 'wb'))

pkl.dump(X_train_farmsize, open('data/X_train_farmsize.np', 'wb'))
pkl.dump(X_val_farmsize, open('data/X_val_farmsize.np', 'wb'))
pkl.dump(X_test_farmsize, open('data/X_test_farmsize.np', 'wb'))

pkl.dump(y_train_gini, open('data/y_train_gini.np', 'wb'))
pkl.dump(y_val_gini, open('data/y_val_gini.np', 'wb'))
pkl.dump(y_test_gini, open('data/y_test_gini.np', 'wb'))

pkl.dump(y_train_tenancy, open('data/y_train_tenancy.np', 'wb'))
pkl.dump(y_val_tenancy, open('data/y_val_tenancy.np', 'wb'))
pkl.dump(y_test_tenancy, open('data/y_test_tenancy.np', 'wb'))

pkl.dump(y_train_farmsize, open('data/y_train_farmsize.np', 'wb'))
pkl.dump(y_val_farmsize, open('data/y_val_farmsize.np', 'wb'))
pkl.dump(y_test_farmsize, open('data/y_test_farmsize.np', 'wb'))