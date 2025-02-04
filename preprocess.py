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
from sklearn.feature_selection import VarianceThreshold

