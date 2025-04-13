# For testing the optimal K values for RF (suggested baseline model per Tima's paper)
    # among feature engineering and feature selection methods

# Synthetic Data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import preprocess
import modeling as models
from metrics import getMetrics

def run_k_test(data, model_name, states, k_vals, title=""):
    methods = ["chi2", "f", "mutual", "pca", "kpca", "umap", "none"]
    selections = ["chi2", "f", "mutual"]
    extractions = ["pca", "kpca", "umap"]

    results = []

    for k in k_vals:
        for method in methods:
            data_splits = {}
            print(f"Testing K = {k}")
            for rst in states:
                data_splits[rst] = preprocess.preprocess(data, rst, selection=method if method in selections else None,
                                                        extraction=method if method in extractions else None, k=k)
            
            model_res = {} # key as model type, then sub dictionary with rst as key and predicted classes & model itself ex: model_res["dt"][rst] --> (y_pred_dt, dt) 
            model_res[model_name] = {}
            func = None
            match model_name: 
                case "rf":
                    func = models.model_randomforest
                case "dt":
                    func = models.model_decisiontree
                case "gb":
                    func = models.model_gradientboosting
                case "xgb":
                    func = models.model_extremegb
                case "lgb":
                    func = models.model_lightgb
                case "et":
                    func = models.model_extratrees
                case "gb":
                    func = models.model_gradientboosting
                case "ab":    
                    func = models.model_adaboost
                case "lr":
                    func = models.model_logisticregression
                case "lr1":
                    func = models.model_lassoregularization
                case "lr2":
                    func = models.model_ridgeRegularization
                case "lre":
                    func = models.model_elasticNetRegularization
                case "lsv":
                    func = models.model_linearSupportVector
                case "nlsv":
                    func = models.model_nonLinearSupportVector
                case "knn":
                    func = models.model_kNearestNeighbor
                case "lda":    
                    func = models.model_linearDiscriminantAnalysis
                case "gnb":    
                    func = models.model_gaussianNaiveBayes
                case "mlp":    
                    func = models.model_multiLayerPerceptron
                case _:
                    raise RuntimeError("Invalid Model Name Received")

            for state in data_splits.keys():    
                model_res[model_name][state] = func(*data_splits[state])

            model_metrics = {}
            model_metrics[model_name] = {}
            model_metrics[model_name]['auc'] = 0.0
            model_metrics[model_name]['acc'] = 0.0
            model_metrics[model_name]['precision'] = 0.0
            model_metrics[model_name]['recall'] = 0.0
            model_metrics[model_name]['f1'] = 0.0

            # calculate the metrics for all models
            for rst in data_splits.keys():
                metric = getMetrics(data_splits[rst][3], data_splits[rst][1], model_res[model_name][rst][0], model_res[model_name][rst][1])
                model_metrics[model_name]['auc'] += metric[0]
                model_metrics[model_name]['acc'] += metric[1]
                model_metrics[model_name]['precision'] += metric[2]
                model_metrics[model_name]['recall'] += metric[3]
                model_metrics[model_name]['f1'] += metric[4]

            # calculate the averages for each model
            model_averages = {}
            model_averages[model_name] = {}
            model_averages[model_name]['auc'] = model_metrics[model_name]['auc'] / len(data_splits)
            model_averages[model_name]['acc'] = model_metrics[model_name]['acc'] / len(data_splits)
            model_averages[model_name]['precision'] = model_metrics[model_name]['precision'] / len(data_splits)
            model_averages[model_name]['recall'] = model_metrics[model_name]['recall'] / len(data_splits)
            model_averages[model_name]['f1'] = model_metrics[model_name]['f1'] / len(data_splits)
            
            print(k, model_averages)
            print(f"Setting: {method}\nK: {k}\nResults: {model_averages}\n\n")
            row = [
             method, k, model_averages[model_name]['auc'],
             model_averages[model_name]['acc'],
             model_averages[model_name]['precision'],
             model_averages[model_name]['recall'],
             model_averages[model_name]['f1']
            ]
            results.append(row)
    df = pd.DataFrame(results, columns=["Method", "K", "AUC", "ACC", "Precision", "Recall", "F1"])

    # Group data by Method
    methods = df['Method'].unique()
    metrics = ["AUC", "ACC", "Precision", "Recall", "F1"]

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        
        for method in methods:
            method_data = df[df['Method'] == method]
            plt.plot(method_data["K"], method_data[metric], label=method, marker='o')

        plt.title(f"{metric} vs K")
        plt.xlabel("K")
        plt.ylabel(metric)
        plt.legend(title="Method")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    return df