import numpy as np
import preprocess
import modeling as models
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, confusion_matrix
from metrics import getMetrics
import matplotlib.pyplot as plt

def main():
    # Synthetic Data
    X, y = make_classification(n_samples=200, n_features=200, random_state=42)
    data = np.concatenate((X, y.reshape(-1, 1)), axis = 1)

    np.random.seed(42)
    states = np.random.randint(low = 0, high = 1000000, size=(100,)) # numpy array with our 100 random states
    data_splits = {}
    for rst in states:
        data_splits[rst] = preprocess.preprocess(data, rst, selection=None, extraction='umap', k=5)
        # k: [1, 10, 20, ..., 120, 130, 140, 150, 160]

    model_names = ["dt", "rf", "gb", "xgb", "lgb", "et", "ab", "lr", "lr1", "lr2", "lre", "lsv", "nlsv", "knn", "lda", "gnb", "mlp"]


    model_res = {} # key as model type, then sub dictionary with rst as key and predicted classes & model itself ex: model_res["dt"][rst] --> (y_pred_dt, dt) 
    for model in model_names:
        model_res[model] = {}

    for state in data_splits.keys():    
        model_res["dt"][state] = models.model_decisiontree(*data_splits[state]) # decision tree
        model_res["rf"][state] = models.model_randomforest(*data_splits[state]) # random forest
        model_res["gb"][state] = models.model_gradientboosting(*data_splits[state]) # grad boosting
        model_res['xgb'][state] = models.model_extremegb(*data_splits[state])
        model_res["lgb"][state] = models.model_lightgb(*data_splits[state])
        model_res["et"][state] = models.model_extratrees(*data_splits[state])
        model_res["ab"][state] = models.model_adaboost(*data_splits[state])
        model_res["lr"][state] = models.model_logisticregression(*data_splits[state])
        model_res["lr1"][state] = models.model_lassoregularization(*data_splits[state])
        model_res["lr2"][state] = models.model_ridgeRegularization(*data_splits[state])
        model_res["lre"][state] = models.model_elasticNetRegularization(*data_splits[state])
        model_res["lsv"][state] = models.model_linearSupportVector(*data_splits[state])
        model_res["nlsv"][state] = models.model_nonLinearSupportVector(*data_splits[state])
        model_res["knn"][state] = models.model_kNearestNeighbor(*data_splits[state])
        model_res["lda"][state] = models.model_linearDiscriminantAnalysis(*data_splits[state])
        model_res["gnb"][state] = models.model_gaussianNaiveBayes(*data_splits[state])
        model_res["mlp"][state] = models.model_multiLayerPerceptron(*data_splits[state])

    # basic view for now to see some metrics
    model_metrics = {}
    for modelName in model_names:
        model_metrics[modelName] = {}
        model_metrics[modelName]['auc'] = 0.0
        model_metrics[modelName]['acc'] = 0.0
        model_metrics[modelName]['precision'] = 0.0
        model_metrics[modelName]['recall'] = 0.0
        model_metrics[modelName]['f1'] = 0.0

    # calculate the metrics for all models
    for rst in data_splits.keys():
        for model in model_names:
            metric = getMetrics(data_splits[rst][3], data_splits[rst][1], model_res[model][rst][0], model_res[model][rst][1])
            model_metrics[model]['auc'] += metric[0]
            model_metrics[model]['acc'] += metric[1]
            model_metrics[model]['precision'] = metric[2]
            model_metrics[model]['recall'] += metric[3]
            model_metrics[model]['f1'] += metric[4]

    # for model in model_names:
        plt.bar(['auc', 'acc', 'precision', 'recall', 'f1'], np.array(list(model_metrics['dt'].values())), color=np.random.rand(3,))
        plt.show()

    # calculate the averages for each model
    model_averages = {}
    for model in model_names:
        model_averages[model] = {}
        model_averages[model]['auc'] = model_metrics[model]['auc'] / len(data_splits)
        model_averages[model]['acc'] = model_metrics[model]['acc'] / len(data_splits)
        model_averages[model]['precision'] = model_metrics[model]['precision'] / len(data_splits)
        model_averages[model]['recall'] = model_metrics[model]['recall'] / len(data_splits)
        model_averages[model]['f1'] = model_metrics[model]['f1'] / len(data_splits)
    print(model_averages)



if __name__ == "__main__":
    main()