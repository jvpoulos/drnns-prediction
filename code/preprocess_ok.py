import pandas as pd
import numpy as np
import cPickle as pkl

# Read data
print('Reading data')

county_x_train = pd.read_csv("data/census-county/county-x-train.csv") # Gini and tenancy
county_x_test = pd.read_csv("data/census-county/county-x-test.csv") 

X_train = county_x_train[county_x_train.columns[3:72]]
X_test = county_x_test[county_x_test.columns[3:72]]

y_train_gini = county_x_train["gini"]
y_test_gini = county_x_test["gini"]

y_train_tenancy = county_x_train["tenancy"]
y_test_tenancy = county_x_test["tenancy"]

print('X_train shape:', X_train.shape)
print('X_val shape:', X_val.shape)
print('X_test shape:', X_test.shape)

# Save train and test sets to disk
print('Save to disk')

pkl.dump(X_train, open('data/X_train.np', 'wb'))
pkl.dump(X_test, open('data/X_test.np', 'wb'))

pkl.dump(y_train_gini, open('data/y_train_gini.np', 'wb'))
pkl.dump(y_test_gini, open('data/y_test_gini.np', 'wb'))

pkl.dump(y_train_tenancy, open('data/y_train_tenancy.np', 'wb'))
pkl.dump(y_test_tenancy, open('data/y_test_tenancy.np', 'wb'))