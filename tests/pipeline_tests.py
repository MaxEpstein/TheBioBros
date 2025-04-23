import pytest
import numpy as np
import sys
from sklearn.datasets import make_classification
import re
sys.path.append("../")
from preprocess import preprocess
from metrics import get_metrics, interpret
import modeling

@pytest.fixture
def array() -> np.ndarray:
    np.random.seed(0)
    return np.random.randint(low=0, high = 100, size=(10,11))

@pytest.fixture
def dataset() -> np.ndarray:
    X, y = make_classification(n_samples=100, random_state=0)
    return np.hstack((X, y.reshape(-1, 1)))

def test_preprocess_invalid_dimensions():
    """Test if given an array with too many dimensions that it will return an exception"""
    array = np.random.rand(10,10,10)
    with pytest.raises(Exception, match="Data must be two-dimensional array. Verify your data is properly shaped."):
        preprocess(array)

def test_preprocess_single_column_array():
    "Tests a single column array raises an exception"
    data = np.random.rand(10, 1)
    with pytest.raises(Exception, match="Data must be two-dimensional array. Verify your data is properly shaped."):
        preprocess(data)

def test_preprocess_invalid_selection_parameter(array: np.ndarray):
    """Test if given a bad selection parameter that it will return a ValueError"""
    with pytest.raises(ValueError, match="Invalid selection parameter."):
        preprocess(array, selection="somethingthatdoesnotexist")

def test_preprocess_invalid_extraction_parameter(array: np.ndarray):
    """Test if given a bad selection parameter that it will return a ValueError"""
    with pytest.raises(ValueError, match="Invalid extraction parameter."):
        preprocess(array, extraction="somethingthatdoesnotexist")

def test_preprocess_invalid_k(array: np.ndarray):
    """Test if given an invalid k that it will return a ValueError"""
    with pytest.raises(ValueError, match="k must be >= 1"):
        preprocess(array, k=-10)

def test_preprocess_empty_array():
    "Tests empty array raises exception"
    data = np.empty((0, 3))
    with pytest.raises(Exception, match="Array contains no rows."):
        preprocess(data)

def test_preprocess_shapes(array: np.ndarray):
    "Tests the preprocessing method returns 80-20 split"
    X_train, X_test, y_train, y_test = preprocess(array, state=0)
    assert X_train.shape[0] == 8
    assert X_test.shape[0] == 2
    assert len(y_train) == 8
    assert len(y_test) == 2

def test_all_zero_variance_features():
    "Tests VarianceThreshold raises an exception if all features have zero variance"
    data = np.zeros(shape=(10,10))  # constant features
    with pytest.raises(Exception, match="No feature in X meets the variance threshold"):
        preprocess(data)

def test_preprocess_scaling(array: np.ndarray):
    "Tests Min-Max scaler ensures all values stay between 0 and 1"
    X_train, _, _, _ = preprocess(array, state=0)
    assert np.all(X_train >= 0) and np.all(X_train <= 1)

def test_preprocess_feature_selection_chi2(dataset: np.ndarray):
    "Tests chi2 functionality"
    X_train, _, _, _ = preprocess(dataset, state=0, selection='chi2', k=2)
    assert X_train.shape[1] == 2

def test_preprocess_feature_selection_f(dataset: np.ndarray):
    "Tests f_classification functionality"
    X_train, _, _, _ = preprocess(dataset, state=0, selection='f', k=2)
    assert X_train.shape[1] == 2

def test_preprocess_feature_selection_mutual(dataset: np.ndarray):
    "Tests mutual information functionality"
    X_train, _, _, _ = preprocess(dataset, state=0, selection='mutual', k=2)
    assert X_train.shape[1] == 2

def test_preprocess_feature_extraction_pca(dataset: np.ndarray):
    "Tests pca functionality"
    X_train, _, _, _ = preprocess(dataset, state=0, extraction='pca', k=2)
    assert X_train.shape[1] == 2

def test_preprocess_feature_extraction_umap(dataset: np.ndarray):
    "Tests umap functionality"
    X_train, _, _, _ = preprocess(dataset, state=0, extraction='umap', k=2)
    assert X_train.shape[1] == 2

def test_preprocess_feature_extraction_kpca(dataset: np.ndarray):
    "Tests kpca functionality"
    X_train, _, _, _ = preprocess(dataset, state=0, extraction='kpca', k=2)
    assert X_train.shape[1] == 2

def test_preprocess_k_larger_than_features(dataset: np.ndarray):
    "Tests that a k larger than the feature amount will return the total number of features in dataset"
    X_train, _, _, _ = preprocess(dataset, state=0, extraction='pca', k=100)
    assert X_train.shape[1] == 20

def test_non_integer_target():
    "Tests labels are integers or integer convertible"
    data = np.array([[i, i+1, "label"] for i in range(10)], dtype=object)
    with pytest.raises(Exception, match="Labels cannot be converted to integer type."):
        preprocess(data)

def test_metrics_get_metrics_basic():
    "Tests basic functionality of get_metrics"
    y_test = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    
    class DummyModel:
        def predict_proba(self, X):
            return np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.6, 0.4]])
    
    model = DummyModel()
    _, accuracy, precision, recall, f1, _, _ = get_metrics(y_test, None, y_pred, model)
    
    assert accuracy == 0.75
    assert precision == 1.0
    assert recall == 0.5
    assert round(f1, 2) == 0.67

def test_metrics_get_metrics_perfect_prediction():
    "Tests that get_metrics properly returns 100% for all metric categories"
    y_test = np.array([0, 1, 1, 0])
    y_pred = y_test.copy()

    class DummyModel:
        def predict_proba(self, X):
            return np.array([[1 - y, y] for y in y_test])
    
    model = DummyModel()
    auc, accuracy, precision, recall, f1, _, _ = get_metrics(y_test, None, y_pred, model)
    
    assert auc == 1.0
    assert accuracy == 1.0
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0

def test_metrics_interpret_skips_unsupported():
    "Ensures models labeled unsupported do not return any data or Exceptions"
    dictionary = {"ab": {}}
    interpret(["ab"], {}, {}, dictionary)
    assert dictionary["ab"]["interpret"] == "unsupported"

@pytest.fixture
def sample_data() -> np.ndarray:
    X, y = make_classification(n_samples=100, n_features=5, random_state=0)
    return X[:80], X[80:], y[:80], y[80:]

@pytest.mark.parametrize("model_func", [
    modeling.model_decisiontree,
    modeling.model_randomforest,
    modeling.model_gradientboosting,
    modeling.model_extremegb,
    modeling.model_lightgb,
    modeling.model_extratrees,
    modeling.model_adaboost,
    modeling.model_logisticregression,
    modeling.model_lassoregularization,
    modeling.model_ridgeRegularization,
    modeling.model_elasticNetRegularization,
    modeling.model_linearSupportVector,
    modeling.model_nonLinearSupportVector,
    modeling.model_kNearestNeighbor,
    modeling.model_linearDiscriminantAnalysis,
    modeling.model_gaussianNaiveBayes,
    modeling.model_multiLayerPerceptron
])

def test_modeling_model_output_shapes(model_func, sample_data: np.ndarray):
    "Ensures that all model fit and predict functions return predictions and the fitted models properly"
    x_train, x_test, y_train, y_test = sample_data
    if ("kNearestNeighbor" in model_func.__name__ or 
    "linearDiscriminantAnalysis" in model_func.__name__ 
    or "gaussianNaiveBayes" in model_func.__name__):
        y_pred, model = model_func(x_train, x_test, y_train, y_test)
    else:
        y_pred, model = model_func(x_train, x_test, y_train, y_test, state=42)
    assert y_pred.shape == y_test.shape, f"{model_func.__name__} prediction shape mismatch"
    assert hasattr(model, "predict"), f"{model_func.__name__} did not return a model"