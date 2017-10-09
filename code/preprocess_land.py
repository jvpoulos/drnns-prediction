import sys
import pandas as pd
import numpy as np
import cPickle as pkl

folder= sys.argv[-1] # 'analysis-12' or 'analysis-34'

## Read data


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

educpc_x_train_treated = pd.read_csv("data/capacity/{}/treated/educpc-x-train.csv".format(folder)) 
educpc_x_val_treated = pd.read_csv("data/capacity/{}/treated/educpc-x-val.csv".format(folder))
educpc_x_test_treated = pd.read_csv("data/capacity/{}/treated/educpc-x-test.csv".format(folder)) 

educpc_y_train_treated = pd.read_csv("data/capacity/{}/treated/educpc-y-train.csv".format(folder)) 
educpc_y_val_treated = pd.read_csv("data/capacity/{}/treated/educpc-y-val.csv".format(folder)) 
educpc_y_test_treated = pd.read_csv("data/capacity/{}/treated/educpc-y-test.csv".format(folder)) 


## Save train and test sets to disk
print('Saving data in data/capacity/{}'.format(folder))

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

pkl.dump(educpc_x_train_treated, open('data/educpc_x_train_treated.np', 'wb')) 
pkl.dump(educpc_x_val_treated, open('data/educpc_x_val_treated.np', 'wb'))
pkl.dump(educpc_x_test_treated, open('data/educpc_x_test_treated.np', 'wb'))

pkl.dump(educpc_y_train_treated, open('data/educpc_y_train_treated.np', 'wb'))
pkl.dump(educpc_y_val_treated, open('data/educpc_y_val_treated.np', 'wb'))
pkl.dump(educpc_y_test_treated, open('data/educpc_y_test_treated.np', 'wb'))