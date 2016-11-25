# Forked from user RenMai (https://www.kaggle.com/hsrobo/d/aaron7sun/stocknews/tf-idf-svm-baseline/code)
# This script concatenates all news headlines of a day into one and uses the tf-idf scheme to extract a feature vector.

# Splits data into training and test set according to competition guidelines (https://www.kaggle.com/aaron7sun/stocknews)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
from datetime import date
import nltk
import cPickle as pkl

# Read data
print('Reading data')

data = pd.read_csv("data/stocknews/Combined_News_DJIA.csv")

# Concatenate all news into one
data["combined_news"] = data.filter(regex=("Top.*")).apply(lambda x: ''.join(str(x.values)), axis=1)

# Split into training, validation, and test set
print('Split data')

training_end = date(2014,12,31)
num_training = len(data[pd.to_datetime(data["Date"]) <= training_end])

X_train_raw = data["combined_news"].values[:num_training]
X_test_raw = data["combined_news"].values[num_training:]

y_train = data["Label"].values[:num_training]
y_test = data["Label"].values[num_training:]

# Feature extraction
print('Extract feature vector')

# Fit tf-idf 
nltk.download(['stopwords', 'wordnet']) # get stopwords from nltk
stopwords = nltk.corpus.stopwords.words('english')

feature_extraction = TfidfVectorizer(stop_words=stopwords,lowercase=True) # remove stopwords and converts characters to lowercase during preprocessing step
X_train = feature_extraction.fit_transform(X_train_raw) # learn vocabulary and idf from training features
X_test = feature_extraction.transform(X_test_raw) # apply training fit to test data

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

# Save train and test sets to disk
print('Save to disk')

pkl.dump(X_train, open('data/X_train.np', 'wb'))
pkl.dump(X_test, open('data/X_test.np', 'wb'))

pkl.dump(y_train, open('data/y_train.np', 'wb'))
pkl.dump(y_test, open('data/y_test.np', 'wb'))