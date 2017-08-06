import sys
import pandas as pd
import numpy as np
import cPickle as pkl

folder= sys.argv[-1] # 'analysis-12' or 'analysis-34'

# Read data
print('Reading data in data/{}'.format(folder))

homesteads_x_train_treated = pd.read_csv("data/patents-public/{}/treated/homesteads-x-train.csv".format(folder)) # Treated
homesteads_x_test_treated = pd.read_csv("data/patents-public/{}/treated/homesteads-x-test.csv".format(folder)) 

homesteads_y_train_treated = pd.read_csv("data/patents-public/{}/treated/homesteads-y-train.csv".format(folder)) 
homesteads_y_test_treated = pd.read_csv("data/patents-public/{}/treated/homesteads-y-test.csv".format(folder)) 

sales_x_train_treated = pd.read_csv("data/patents-public/{}/treated/sales-x-train.csv".format(folder)) 
sales_x_test_treated = pd.read_csv("data/patents-public/{}/treated/sales-x-test.csv".format(folder)) 

sales_y_train_treated = pd.read_csv("data/patents-public/{}/treated/sales-y-train.csv".format(folder)) 
sales_y_test_treated = pd.read_csv("data/patents-public/{}/treated/sales-y-test.csv".format(folder)) 

homesteads_x_train_control = pd.read_csv("data/patents-public/{}/control/homesteads-x-train.csv".format(folder)) # Control
homesteads_x_test_control = pd.read_csv("data/patents-public/{}/control/homesteads-x-test.csv".format(folder)) 

homesteads_y_train_control = pd.read_csv("data/patents-public/{}/control/homesteads-y-train.csv".format(folder)) 
homesteads_y_test_control = pd.read_csv("data/patents-public/{}/control/homesteads-y-test.csv".format(folder)) 

sales_x_train_control = pd.read_csv("data/patents-public/{}/control/sales-x-train.csv".format(folder)) 
sales_x_test_control = pd.read_csv("data/patents-public/{}/control/sales-x-test.csv".format(folder)) 

sales_y_train_control = pd.read_csv("data/patents-public/{}/control/sales-y-train.csv".format(folder)) 
sales_y_test_control = pd.read_csv("data/patents-public/{}/control/sales-y-test.csv".format(folder)) 


# Save train and test sets to disk
print('Save to disk')

pkl.dump(homesteads_x_train_treated, open('data/homesteads_x_train_treated.np', 'wb')) # Treated
pkl.dump(homesteads_x_test_treated, open('data/homesteads_x_test_treated.np', 'wb'))

pkl.dump(sales_x_train_treated, open('data/sales_x_train_treated.np', 'wb'))
pkl.dump(sales_x_test_treated, open('data/sales_x_test_treated.np', 'wb'))

pkl.dump(homesteads_y_train_treated, open('data/homesteads_y_train_treated.np', 'wb'))
pkl.dump(homesteads_y_test_treated, open('data/homesteads_y_test_treated.np', 'wb'))

pkl.dump(sales_y_train_treated, open('data/sales_y_train_treated.np', 'wb'))
pkl.dump(sales_y_test_treated, open('data/sales_y_test_treated.np', 'wb'))

pkl.dump(homesteads_x_train_control, open('data/homesteads_x_train_control.np', 'wb')) # Control
pkl.dump(homesteads_x_test_control, open('data/homesteads_x_test_control.np', 'wb'))

pkl.dump(sales_x_train_control, open('data/sales_x_train_control.np', 'wb'))
pkl.dump(sales_x_test_control, open('data/sales_x_test_control.np', 'wb'))

pkl.dump(homesteads_y_train_control, open('data/homesteads_y_train_control.np', 'wb'))
pkl.dump(homesteads_y_test_control, open('data/homesteads_y_test_control.np', 'wb'))

pkl.dump(sales_y_train_control, open('data/sales_y_train_control.np', 'wb'))
pkl.dump(sales_y_test_control, open('data/sales_y_test_control.np', 'wb'))

