import pandas as pd
import numpy as np
import cPickle as pkl

# Read data
print('Reading data')

homesteads_x_train = pd.read_csv("data/patents/homesteads-x-train.csv") 
homesteads_x_test = pd.read_csv("data/patents/homesteads-x-test.csv") 

homesteads_y_train = pd.read_csv("data/patents/homesteads-y-train.csv") 
homesteads_y_test = pd.read_csv("data/patents/homesteads-y-test.csv") 

sales_x_train = pd.read_csv("data/patents/sales-x-train.csv") 
sales_x_test = pd.read_csv("data/patents/sales-x-test.csv") 

sales_y_train = pd.read_csv("data/patents/sales-y-train.csv") 
sales_y_test = pd.read_csv("data/patents/sales-y-test.csv") 

homesteads_x_train_placebo = pd.read_csv("data/patents/homesteads-x-train-placebo.csv")  # placebos
homesteads_x_test_placebo = pd.read_csv("data/patents/homesteads-x-test-placebo.csv") 

homesteads_y_train_placebo = pd.read_csv("data/patents/homesteads-y-train-placebo.csv") 
homesteads_y_test_placebo = pd.read_csv("data/patents/homesteads-y-test-placebo.csv") 

sales_x_train_placebo = pd.read_csv("data/patents/sales-x-train-placebo.csv") 
sales_x_test_placebo = pd.read_csv("data/patents/sales-x-test-placebo.csv") 

sales_y_train_placebo = pd.read_csv("data/patents/sales-y-train-placebo.csv") 
sales_y_test_placebo = pd.read_csv("data/patents/sales-y-test-placebo.csv") 


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

pkl.dump(homesteads_x_train_placebo, open('data/homesteads_x_train_placebo.np', 'wb')) # placebos
pkl.dump(homesteads_x_test_placebo, open('data/homesteads_x_test_placebo.np', 'wb'))

pkl.dump(sales_x_train_placebo, open('data/sales_x_train_placebo.np', 'wb'))
pkl.dump(sales_x_test_placebo, open('data/sales_x_test_placebo.np', 'wb'))

pkl.dump(homesteads_y_train_placebo, open('data/homesteads_y_train_placebo.np', 'wb'))
pkl.dump(homesteads_y_test_placebo, open('data/homesteads_y_test_placebo.np', 'wb'))

pkl.dump(sales_y_train_placebo, open('data/sales_y_train_placebo.np', 'wb'))
pkl.dump(sales_y_test_placebo, open('data/sales_y_test_placebo.np', 'wb'))