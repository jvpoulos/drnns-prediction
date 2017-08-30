import sys
import pandas as pd
import numpy as np
import cPickle as pkl

folder= sys.argv[-1] # 'analysis-12' or 'analysis-34'

## Read data

# patents
print('Reading data in data/patents-public/{}'.format(folder))

homesteads_x_train_treated = pd.read_csv("data/patents-public/{}/treated/homesteads-x-train.csv".format(folder)) # Treated
homesteads_x_test_treated = pd.read_csv("data/patents-public/{}/treated/homesteads-x-test.csv".format(folder)) 
homesteads_x_val_treated = pd.read_csv("data/patents-public/{}/treated/homesteads-x-val.csv".format(folder)) 

homesteads_y_train_treated = pd.read_csv("data/patents-public/{}/treated/homesteads-y-train.csv".format(folder)) 
homesteads_y_test_treated = pd.read_csv("data/patents-public/{}/treated/homesteads-y-test.csv".format(folder)) 
homesteads_y_val_treated = pd.read_csv("data/patents-public/{}/treated/homesteads-y-val.csv".format(folder)) 

sales_x_train_treated = pd.read_csv("data/patents-public/{}/treated/sales-x-train.csv".format(folder)) 
sales_x_test_treated = pd.read_csv("data/patents-public/{}/treated/sales-x-test.csv".format(folder)) 
sales_x_val_treated = pd.read_csv("data/patents-public/{}/treated/sales-x-val.csv".format(folder)) 

sales_y_train_treated = pd.read_csv("data/patents-public/{}/treated/sales-y-train.csv".format(folder)) 
sales_y_test_treated = pd.read_csv("data/patents-public/{}/treated/sales-y-test.csv".format(folder)) 
sales_y_val_treated = pd.read_csv("data/patents-public/{}/treated/sales-y-val.csv".format(folder)) 

# capacity
print('Reading data in data/capacity/{}'.format(folder))

revpc_x_train_treated = pd.read_csv("data/capacity/{}/treated/revpc-x-train.csv".format(folder)) 
revpc_x_val_treated = pd.read_csv("data/capacity/{}/treated/revpc-x-val.csv".format(folder))
revpc_x_test_treated = pd.read_csv("data/capacity/{}/treated/revpc-x-test.csv".format(folder)) 

revpc_y_train_treated = pd.read_csv("data/capacity/{}/treated/revpc-y-train.csv".format(folder)) 
revpc_y_val_treated = pd.read_csv("data/capacity/{}/treated/revpc-y-val.csv".format(folder)) 
revpc_y_test_treated = pd.read_csv("data/capacity/{}/treated/revpc-y-test.csv".format(folder)) 

exppc_x_train_treated = pd.read_csv("data/capacity/{}/treated/exppc-x-train.csv".format(folder)) 
exppc_x_val_treated = pd.read_csv("data/capacity/{}/treated/exppc-x-val.csv".format(folder))
exppc_x_test_treated = pd.read_csv("data/capacity/{}/treated/exppc-x-test.csv".format(folder)) 

exppc_y_train_treated = pd.read_csv("data/capacity/{}/treated/exppc-y-train.csv".format(folder)) 
exppc_y_val_treated = pd.read_csv("data/capacity/{}/treated/exppc-y-val.csv".format(folder)) 
exppc_y_test_treated = pd.read_csv("data/capacity/{}/treated/exppc-y-test.csv".format(folder)) 

edpc_x_train_treated = pd.read_csv("data/capacity/{}/treated/edpc-x-train.csv".format(folder)) 
edpc_x_val_treated = pd.read_csv("data/capacity/{}/treated/edpc-x-val.csv".format(folder)) 
edpc_x_test_treated = pd.read_csv("data/capacity/{}/treated/edpc-x-test.csv".format(folder)) 

edpc_y_train_treated = pd.read_csv("data/capacity/{}/treated/edpc-y-train.csv".format(folder)) 
edpc_y_val_treated = pd.read_csv("data/capacity/{}/treated/edpc-y-val.csv".format(folder)) 
edpc_y_test_treated = pd.read_csv("data/capacity/{}/treated/edpc-y-test.csv".format(folder)) 

## Save train and test sets to disk
print('Save to disk')

# patents
pkl.dump(homesteads_x_train_treated, open('data/homesteads_x_train_treated.np', 'wb')) 
pkl.dump(homesteads_x_val_treated, open('data/homesteads_x_val_treated.np', 'wb'))
pkl.dump(homesteads_x_test_treated, open('data/homesteads_x_test_treated.np', 'wb'))

pkl.dump(sales_x_train_treated, open('data/sales_x_train_treated.np', 'wb'))
pkl.dump(sales_x_val_treated, open('data/sales_x_val_treated.np', 'wb'))
pkl.dump(sales_x_test_treated, open('data/sales_x_test_treated.np', 'wb'))

pkl.dump(homesteads_y_train_treated, open('data/homesteads_y_train_treated.np', 'wb'))
pkl.dump(homesteads_y_val_treated, open('data/homesteads_y_val_treated.np', 'wb'))
pkl.dump(homesteads_y_test_treated, open('data/homesteads_y_test_treated.np', 'wb'))

pkl.dump(sales_y_train_treated, open('data/sales_y_train_treated.np', 'wb'))
pkl.dump(sales_y_val_treated, open('data/sales_y_val_treated.np', 'wb'))
pkl.dump(sales_y_test_treated, open('data/sales_y_test_treated.np', 'wb'))

# capacity
pkl.dump(revpc_x_train_treated, open('data/revpc_x_train_treated.np', 'wb')) 
pkl.dump(revpc_x_val_treated, open('data/revpc_x_val_treated.np', 'wb'))
pkl.dump(revpc_x_test_treated, open('data/revpc_x_test_treated.np', 'wb'))

pkl.dump(revpc_y_train_treated, open('data/revpc_y_train_treated.np', 'wb'))
pkl.dump(revpc_y_val_treated, open('data/revpc_y_val_treated.np', 'wb'))
pkl.dump(revpc_y_test_treated, open('data/revpc_y_test_treated.np', 'wb'))

pkl.dump(exppc_x_train_treated, open('data/exppc_x_train_treated.np', 'wb')) 
pkl.dump(exppc_x_val_treated, open('data/exppc_x_val_treated.np', 'wb'))
pkl.dump(exppc_x_test_treated, open('data/exppc_x_test_treated.np', 'wb'))

pkl.dump(exppc_y_train_treated, open('data/exppc_y_train_treated.np', 'wb'))
pkl.dump(exppc_y_val_treated, open('data/exppc_y_val_treated.np', 'wb'))
pkl.dump(exppc_y_test_treated, open('data/exppc_y_test_treated.np', 'wb'))

pkl.dump(edpc_x_train_treated, open('data/edpc_x_train_treated.np', 'wb'))
pkl.dump(edpc_x_val_treated, open('data/edpc_x_val_treated.np', 'wb'))
pkl.dump(edpc_x_test_treated, open('data/edpc_x_test_treated.np', 'wb'))

pkl.dump(edpc_y_train_treated, open('data/edpc_y_train_treated.np', 'wb'))
pkl.dump(edpc_y_val_treated, open('data/edpc_y_val_treated.np', 'wb'))
pkl.dump(edpc_y_test_treated, open('data/edpc_y_test_treated.np', 'wb'))