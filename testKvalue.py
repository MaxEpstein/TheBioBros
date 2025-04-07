# For testing the optimal K values for RF (suggested baseline model per Tima's paper)
    # among feature engineering and feature selection methods

k_vals = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]

methods = ["chi2", "f", "mutual", "pca", "kpca", "umap", None]
selections = ["chi2", "f", "mutual"]
extractions = ["pca", "kpca", "umap"]

# Synthetic Data
import numpy as np
import preprocess
import modeling as models
from sklearn.datasets import make_classification
from metrics import getMetrics

X, y = make_classification(n_samples=200, n_features=200, random_state=42)
data = np.concatenate((X, y.reshape(-1, 1)), axis = 1)
np.random.seed(42)
states = np.random.randint(low = 0, high = 1000000, size=(100,)) # numpy array with our 100 random states

with open('k_test.txt', mode="w") as file:
    for k in k_vals:
        for method in methods:
            data_splits = {}
            print(f"Testing K = {k}")
            for rst in states:
                data_splits[rst] = preprocess.preprocess(data, rst, selection=method if method in selections else None,
                                                        extraction=method if method in extractions else None, k=k)
            
            model_res = {} # key as model type, then sub dictionary with rst as key and predicted classes & model itself ex: model_res["dt"][rst] --> (y_pred_dt, dt) 
            model_res["rf"] = {}

            for state in data_splits.keys():    
                model_res["rf"][state] = models.model_randomforest(*data_splits[state])

            model_metrics = {}
            model_metrics["rf"] = {}
            model_metrics["rf"]['auc'] = 0.0
            model_metrics["rf"]['acc'] = 0.0
            model_metrics["rf"]['precision'] = 0.0
            model_metrics["rf"]['recall'] = 0.0
            model_metrics["rf"]['f1'] = 0.0

            # calculate the metrics for all models
            for rst in data_splits.keys():
                metric = getMetrics(data_splits[rst][3], data_splits[rst][1], model_res["rf"][rst][0], model_res["rf"][rst][1])
                model_metrics["rf"]['auc'] += metric[0]
                model_metrics["rf"]['acc'] += metric[1]
                model_metrics["rf"]['precision'] += metric[2]
                model_metrics["rf"]['recall'] += metric[3]
                model_metrics["rf"]['f1'] += metric[4]

            # calculate the averages for each model
            model_averages = {}
            model_averages["rf"] = {}
            model_averages["rf"]['auc'] = model_metrics["rf"]['auc'] / len(data_splits)
            model_averages["rf"]['acc'] = model_metrics["rf"]['acc'] / len(data_splits)
            model_averages["rf"]['precision'] = model_metrics["rf"]['precision'] / len(data_splits)
            model_averages["rf"]['recall'] = model_metrics["rf"]['recall'] / len(data_splits)
            model_averages["rf"]['f1'] = model_metrics["rf"]['f1'] / len(data_splits)
            print(k, model_averages)
            file.write(f"Setting: {method}\nK: {k}\nResults: {model_averages}\n\n")
