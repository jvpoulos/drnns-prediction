import pandas as pd
import numpy as np
import cPickle as pkl

# Read data
print('Reading data')

gini_x_train = pd.read_csv("data/census/gini-x-train.csv") 
gini_x_test = pd.read_csv("data/census/gini-x-test.csv") 

gini_y_train = pd.read_csv("data/census/gini-y-train.csv") 
gini_y_test = pd.read_csv("data/census/gini-y-test.csv") 

tenancy_x_train = pd.read_csv("data/census/tenancy-x-train.csv") 
tenancy_x_test = pd.read_csv("data/census/tenancy-x-test.csv") 

tenancy_y_train = pd.read_csv("data/census/tenancy-y-train.csv") 
tenancy_y_test = pd.read_csv("data/census/tenancy-y-test.csv") 

gini_x_train_placebo = pd.read_csv("data/census/gini-x-train-placebo.csv") 
gini_x_val_placebo = pd.read_csv("data/census/gini-x-val-placebo.csv") 

gini_y_train_placebo = pd.read_csv("data/census/gini-y-train-placebo.csv") 
gini_y_val_placebo = pd.read_csv("data/census/gini-y-val-placebo.csv") 

tenancy_x_train_placebo = pd.read_csv("data/census/tenancy-x-train-placebo.csv") 
tenancy_x_val_placebo = pd.read_csv("data/census/tenancy-x-val-placebo.csv") 

tenancy_y_train_placebo = pd.read_csv("data/census/tenancy-y-train-placebo.csv") 
tenancy_y_val_placebo = pd.read_csv("data/census/tenancy-y-val-placebo.csv") 


# Save train and test sets to disk
print('Save to disk')

pkl.dump(gini_x_train, open('data/gini_x_train.np', 'wb'))
pkl.dump(gini_x_test, open('data/gini_x_test.np', 'wb'))

pkl.dump(tenancy_x_train, open('data/tenancy_x_train.np', 'wb'))
pkl.dump(tenancy_x_test, open('data/tenancy_x_test.np', 'wb'))

pkl.dump(gini_y_train, open('data/gini_y_train.np', 'wb'))
pkl.dump(gini_y_test, open('data/gini_y_test.np', 'wb'))

pkl.dump(tenancy_y_train, open('data/tenancy_y_train.np', 'wb'))
pkl.dump(tenancy_y_test, open('data/tenancy_y_test.np', 'wb'))

pkl.dump(gini_x_train_placebo, open('data/gini_x_train_placebo.np', 'wb'))
pkl.dump(gini_x_val_placebo, open('data/gini_x_val_placebo.np', 'wb'))

pkl.dump(tenancy_x_train_placebo, open('data/tenancy_x_train_placebo.np', 'wb'))
pkl.dump(tenancy_x_val_placebo, open('data/tenancy_x_val_placebo.np', 'wb'))

pkl.dump(gini_y_train_placebo, open('data/gini_y_train_placebo.np', 'wb'))
pkl.dump(gini_y_val_placebo, open('data/gini_y_val_placebo.np', 'wb'))

pkl.dump(tenancy_y_train_placebo, open('data/tenancy_y_train_placebo.np', 'wb'))
pkl.dump(tenancy_y_val_placebo, open('data/tenancy_y_val_placebo.np', 'wb'))