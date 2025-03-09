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
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif, SelectKBest
from sklearn.decomposition import PCA, KernelPCA
# import umap # <-- dependency issue needs to be solved

def preprocess(data, state=42, selection=None, extraction=None, k=10):
    try:
        assert len(data.shape) == 2
        assert data.shape[1] >= 2
    except:
        raise Exception("Data must be two-dimensional array. Verify your data is properly shaped.")
    
    X, y = data[:,:-1], data[:,-1]
    y = y.astype(int)
    
    # 1b - Encode Target Variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2 - Remove features with zero variance (VarianceThreshold)
    vt = VarianceThreshold()
    X_train = vt.fit_transform(X_train)
    X_test = vt.transform(X_test)

    # 3 - Normalize the dataset (MinMax Scaler)
    mm = MinMaxScaler()
    X_train = mm.fit_transform(X_train)
    X_test = mm.transform(X_test)

    # 4 - Feature Selection
    if selection:
        match selection:
            case 'chi2': 
                selector = SelectKBest(chi2, k=min(k, X_train.shape[1]))
            case 'f':
                selector = SelectKBest(f_classif, k=min(k, X_train.shape[1]))
            case 'mutual':
                selector = SelectKBest(mutual_info_classif, k=min(k, X_train.shape[1]))
            case None:
                selector = None
            case _:
                raise ValueError("Invalid selection parameter.")
        
        if selector:
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)
    

    # 5 - Feature Engineering
    if extraction:
        match extraction:
            case 'pca': 
                extractor = PCA(n_components=min(k, X_train.shape[1]), random_state=state)
            case 'kpca':
                extractor = KernelPCA(n_components=min(k, X_train.shape[1]), kernel="rbf")
            case 'umap': # Need to revisit dependencies to fix!!!
                extractor = None
                # extractor = umap.UMAP(n_components=min(k, X_train.shape[1]), random_state=state) 
            case None:
                extractor = None
            case _:
                raise ValueError("Invalid extraction parameter.")
        
        if extractor:
            X_train = extractor.fit_transform(X_train)
            X_test = extractor.transform(X_test)

    return (X_train, X_test, y_train, y_test)