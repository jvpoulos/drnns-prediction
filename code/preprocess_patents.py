import pandas as pd
import numpy as np
import cPickle as pkl

# Read data
print('Reading data')

homesteads_x_train = pd.read_csv("data/patents/homesteads_x_train.csv") 
homesteads_x_test = pd.read_csv("data/patents/homesteads_x_test.csv") 

homesteads_y_train = pd.read_csv("data/patents/homesteads_y_train.csv") 
homesteads_y_test = pd.read_csv("data/patents/homesteads_y_test.csv") 

sales_x_train = pd.read_csv("data/patents/sales_x_train.csv") 
sales_x_test = pd.read_csv("data/patents/sales_x_test.csv") 

sales_y_train = pd.read_csv("data/patents/sales_y_train.csv") 
sales_y_test = pd.read_csv("data/patents/sales_y_test.csv") 


# Save train and test sets to disk
print('Save to disk')

pkl.dump(homesteads_x_train, open('data/homesteads_x_train.np', 'wb'))
pkl.dump(homesteads_x_test, open('data/homesteads_x_test.np', 'wb'))

pkl.dump(sales_x_train, open('data/sales_x_train.np', 'wb'))
pkl.dump(sales_x_test, open('data/sales_x_test.np', 'wb'))

pkl.dump(homesteads_y_train, open('data/homesteads_y_train.np', 'wb'))
pkl.dump(homesteads_y_test, open('data/homesteads_y_test.np', 'wb'))

pkl.dump(sales_y_train, open('data/sales_y_train.np', 'wb'))
pkl.dump(sales_y_test, open('data/sales_y_test.np', 'wb'))