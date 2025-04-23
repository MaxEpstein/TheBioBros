# This file is for the preprocessing component of the pipeline.
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif, SelectKBest
from sklearn.decomposition import PCA, KernelPCA
import umap

def preprocess(data, state=42, selection=None, extraction=None, k=10):
    '''Preprocesses datasets by encoding the target variable,
    doing a train-test split of 80% - 20%, removing features with zero variance,
    normalizing via Min-Max Scaler, and optionally applying feature selection or extraction.
    
    Args:
    data: the dataset we wish to preprocess. Must be numeric and labels must be integers 1 or 0
    state: random state for splits and optional feature engineering methods
    selection: optional selection of a feature selection method (ex: 'chi2', 'f', 'mutual', None)
    extraction: optional selection of a feature extraction method (ex: 'pca', 'kpca', 'umap', None)
    k: corresponds to k best features for selection and k components for extraction. 
        Must be >= 1 and if greater than number of features it will be set to number of features
    
    Return:
    tuple of (X_train, X_test, y_train, y_test)
    '''
    try:
        assert len(data.shape) == 2
        assert data.shape[1] >= 2
    except:
        raise Exception("Data must be two-dimensional array. Verify your data is properly shaped.")
    
    try:
        assert k >= 1
    except:
        raise ValueError("k must be >= 1")
    
    try:
        assert data.shape[0] > 0
    except:
        raise Exception("Array contains no rows.")

    
    X, y = data[:,:-1], data[:,-1]
    try:
        y = y.astype(int)
    except:
        raise Exception("Labels cannot be converted to integer type.")
    
    # 1b - Encode Target Variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=state)
    
    # 2 - Remove features with zero variance (VarianceThreshold)
    try:
        vt = VarianceThreshold()
        X_train = vt.fit_transform(X_train)
        X_test = vt.transform(X_test)
    except:
        raise Exception("No feature in X meets the variance threshold")
        

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
                if k >= X_train.shape[1]:
                    print(f"k value too large, using {X_train.shape[1] - 1} instead.")
                extractor = umap.UMAP(n_components=min(k, X_train.shape[1] - 1), random_state=state) 
            case None:
                extractor = None
            case _:
                raise ValueError("Invalid extraction parameter.")
        
        if extractor:
            X_train = extractor.fit_transform(X_train)
            X_test = extractor.transform(X_test)

    return (X_train, X_test, y_train, y_test)