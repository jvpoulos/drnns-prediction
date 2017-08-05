import pandas as pd
import numpy as np
import cPickle as pkl

# Read data
print('Reading data')

homesteads_x_train_treated = pd.read_csv("data/patents-public/treated/homesteads-x-train.csv") # Analysis 1
homesteads_x_test_treated = pd.read_csv("data/patents-public/treated/homesteads-x-test.csv") 

homesteads_y_train_treated = pd.read_csv("data/patents-public/treated/homesteads-y-train.csv") 
homesteads_y_test_treated = pd.read_csv("data/patents-public/treated/homesteads-y-test.csv") 

sales_x_train_treated = pd.read_csv("data/patents-public/treated/sales-x-train.csv") 
sales_x_test_treated = pd.read_csv("data/patents-public/treated/sales-x-test.csv") 

sales_y_train_treated = pd.read_csv("data/patents-public/treated/sales-y-train.csv") 
sales_y_test_treated = pd.read_csv("data/patents-public/treated/sales-y-test.csv") 

homesteads_x_train_control = pd.read_csv("data/patents-public/control/homesteads-x-train.csv") # Analysis 2
homesteads_x_test_control = pd.read_csv("data/patents-public/control/homesteads-x-test.csv") 

homesteads_y_train_control = pd.read_csv("data/patents-public/control/homesteads-y-train.csv") 
homesteads_y_test_control = pd.read_csv("data/patents-public/control/homesteads-y-test.csv") 

sales_x_train_control = pd.read_csv("data/patents-public/control/sales-x-train.csv") 
sales_x_test_control = pd.read_csv("data/patents-public/control/sales-x-test.csv") 

sales_y_train_control = pd.read_csv("data/patents-public/control/sales-y-train.csv") 
sales_y_test_control = pd.read_csv("data/patents-public/control/sales-y-test.csv") 


# Save train and test sets to disk
print('Save to disk')

pkl.dump(homesteads_x_train_treated, open('data/homesteads_x_train_treated.np', 'wb')) # Analysis 1
pkl.dump(homesteads_x_test_treated, open('data/homesteads_x_test_treated.np', 'wb'))

pkl.dump(sales_x_train_treated, open('data/sales_x_train_treated.np', 'wb'))
pkl.dump(sales_x_test_treated, open('data/sales_x_test_treated.np', 'wb'))

pkl.dump(homesteads_y_train_treated, open('data/homesteads_y_train_treated.np', 'wb'))
pkl.dump(homesteads_y_test_treated, open('data/homesteads_y_test_treated.np', 'wb'))

pkl.dump(sales_y_train_treated, open('data/sales_y_train_treated.np', 'wb'))
pkl.dump(sales_y_test_treated, open('data/sales_y_test_treated.np', 'wb'))

pkl.dump(homesteads_x_train_control, open('data/homesteads_x_train_control.np', 'wb')) # Analysis 2
pkl.dump(homesteads_x_test_control, open('data/homesteads_x_test_control.np', 'wb'))

pkl.dump(sales_x_train_control, open('data/sales_x_train_control.np', 'wb'))
pkl.dump(sales_x_test_control, open('data/sales_x_test_control.np', 'wb'))

pkl.dump(homesteads_y_train_control, open('data/homesteads_y_train_control.np', 'wb'))
pkl.dump(homesteads_y_test_control, open('data/homesteads_y_test_control.np', 'wb'))

pkl.dump(sales_y_train_control, open('data/sales_y_train_control.np', 'wb'))
pkl.dump(sales_y_test_control, open('data/sales_y_test_control.np', 'wb'))

