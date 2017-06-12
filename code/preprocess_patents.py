import pandas as pd
import numpy as np
import cPickle as pkl

# Read data
print('Reading data')

patents_train = pd.read_csv("data/patents/patents-train.csv") 
patents_val = pd.read_csv("data/patents/patents-val.csv")
patents_test = pd.read_csv("data/patents/patents-test.csv")

X_train = patents_train[patents_train.columns[3:15]]
X_val = patents_val[patents_val.columns[3:15]]
X_test = patents_test[patents_test.columns[3:15]]

y_train_sales = patents_train["sales"]
y_val_sales = patents_val["sales"]
y_test_sales = patents_test["sales"]

y_train_homesteads = patents_train["homesteads"]
y_val_homesteads = patents_val["homesteads"]
y_test_homesteads = patents_test["homesteads"]

print('X_train shape:', X_train.shape)
print('X_val shape:', X_val.shape)
print('X_test shape:', X_test.shape)

# Save train and test sets to disk
print('Save to disk')

pkl.dump(X_train, open('data/X_train_patents.np', 'wb'))
pkl.dump(X_val, open('data/X_val_patents.np', 'wb'))
pkl.dump(X_test, open('data/X_test_patents.np', 'wb'))

pkl.dump(y_train_sales, open('data/y_train_sales.np', 'wb'))
pkl.dump(y_val_sales, open('data/y_val_sales.np', 'wb'))
pkl.dump(y_test_sales, open('data/y_test_sales.np', 'wb'))

pkl.dump(y_train_homesteads, open('data/y_train_homesteads.np', 'wb'))
pkl.dump(y_val_homesteads, open('data/y_val_homesteads.np', 'wb'))
pkl.dump(y_test_homesteads, open('data/y_test_homesteads.np', 'wb'))