# DRNNs-prediction

Implements deep RNNs using the [Keras](https://keras.io/) neural networks library on [Daily News for Stock Market Prediction](https://www.kaggle.com/aaron7sun/stocknews) dataset from [Kaggle](https://www.kaggle.com/).

The dataset task is to predict future movement of the DJIA using current and previous days' news headlines as features.

#Code
* _preprocess.py_ concatenates all news headlines of a day into one and uses the tf-idf scheme to extract a feature vector. Saves preprocessed training and test data sets to disk. 
* _baseline.py_ trains a 3-layer "stateful" RNN and evalates on test set in terms of AUC and classification accuracy. Returns ROC curve plot and image visualization of model. 
* _final_train.py_ trains a 12-layer "stateful" RNN and saves weights at each epoch and epoch results to a .csv file.
* _final_predict.py_ Loads best weights and makes predictions on test set. Evalates on test set in terms of AUC and classification accuracy. Returns ROC curve plot and image visualization of model. Takes weight file location as command-line argument, e.g., `python code/final_predict.py "results/weights/weights-0.5423.hdf5"`
* _plot_history.py_ plots training/test loss and classification accuracy vs. epochs. Takes epoch results as command-line argument, e.g., `python code/plot_history.py "/Users/jason/Dropbox/github/DRNNs-prediction/results/training_log.csv"`
* _model_tex.py_ converts dot object visualization of model to tex/tikz format
