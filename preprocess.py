# This file is for the preprocessing component of the pipeline.

# Steps:
    # 1a: Encode Target Variable
        # 1b: Train-Test Split of 80% - 20%
    # 2: Remove features with zero variance (VarianceThreshold)
    # 3: Normalize the dataset (MinMax Scaler)
        # NOTE: Do Steps 2 and 3 by fitting to Training Set, then applying same transformation to test set also 
            # (but don't train on test set!!!)

# Imports
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

def preprocess(data, state=42):
    X, y = data[:,:-1], data[:,-1]
    y = y.astype(int)
    
    # 1b - Encode Target Variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42) # TODO: reconfigure to run 100 times and save state each time
    
    # 2 - Remove features with zero variance (VarianceThreshold)
    vt = VarianceThreshold()
    X_train = vt.fit_transform(X_train)
    X_test = vt.transform(X_test)

    # 3 - Normalize the dataset (MinMax Scaler)
    mm = MinMaxScaler()
    X_train = mm.fit_transform(X_train)
    X_test = mm.transform(X_test)

    return (X_train, X_test, y_train, y_test)