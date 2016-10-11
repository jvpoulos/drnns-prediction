# Forked from user RenMai (https://www.kaggle.com/hsrobo/d/aaron7sun/stocknews/tf-idf-svm-baseline/code)
# This script concatenates all news headlines of a day into one and uses the tf-idf scheme to extract a feature vector.

# Splits data into training and test set according to competition guidelines (https://www.kaggle.com/aaron7sun/stocknews)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
from datetime import date
import cPickle as pkl

def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

# Read data
data = pd.read_csv("data/stocknews/Combined_News_DJIA.csv")

# Concatenate all news into one
data["combined_news"] = data.filter(regex=("Top.*")).apply(lambda x: ''.join(str(x.values)), axis=1)

# Convert to feature vector
feature_extraction = TfidfVectorizer() # converts all characters to lowercase by default
X = feature_extraction.fit_transform(data["combined_news"].values)

# Split into training, validation, and test set
training_end = date(2014,12,31)
num_training = len(data[pd.to_datetime(data["Date"]) <= training_end])

X_train = X[:num_training]
X_test = X[num_training:]

y_train = data["Label"].values[:num_training]
y_test = data["Label"].values[num_training:]

# Save train and test sets to disk
pkl.dump(X_train, open('data/X_train.np', 'wb'))
pkl.dump(X_test, open('data/X_test.np', 'wb'))

pkl.dump(y_train, open('data/y_train.np', 'wb'))
pkl.dump(y_test, open('data/y_test.np', 'wb'))