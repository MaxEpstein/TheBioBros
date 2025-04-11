import numpy as np
import preprocess
import modeling as models
from metrics import getMetrics
import pandas as pd
from metrics import interpret
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_LIST = ["dt", "rf", "gb", "xgb", "lgb", "et","ab", "lr", "lr1", "lr2", "lre", "lsv", "nlsv", "knn", "lda", "gnb", "mlp"]

def pipeline(data, selection=None, extraction=None, k=20):
    np.random.seed(42)
    states = np.random.randint(low = 0, high = 1000000, size=(100,))
    data_splits = {}
    for rst in states:
        data_splits[rst] = preprocess.preprocess(data, rst, selection=selection, extraction=extraction, k=k) # TODO: figure out about k
    
    model_res = {}
    for model in MODEL_LIST:
        model_res[model] = {}

    for state in data_splits.keys():    
        model_res["dt"][state] = models.model_decisiontree(*data_splits[state])
        model_res["rf"][state] = models.model_randomforest(*data_splits[state])
        model_res["gb"][state] = models.model_gradientboosting(*data_splits[state])
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
        print("Finished training models on round: " + state.astype(str))

    model_metrics = {}
    for model in MODEL_LIST:
        model_metrics[model] = {}

    # calculate the metrics for all models
    for rst in data_splits.keys():
        for model in MODEL_LIST:
            model_metrics[model][rst] = {}
            metric = getMetrics(data_splits[rst][3], data_splits[rst][1], model_res[model][rst][0], model_res[model][rst][1])
            model_metrics[model][rst]['auc'] = metric[0]
            model_metrics[model][rst]['acc'] = metric[1]
            model_metrics[model][rst]['precision'] = metric[2]
            model_metrics[model][rst]['recall'] = metric[3]
            model_metrics[model][rst]['f1'] = metric[4]
    
    model_averages = {}
    for model in MODEL_LIST:
        model_averages[model] = {}
        model_averages[model]['auc'] = np.mean([model_metrics[model][rst]['auc'] for rst in data_splits.keys()])
        model_averages[model]['acc'] = np.mean([model_metrics[model][rst]['acc'] for rst in data_splits.keys()])
        model_averages[model]['precision'] = np.mean([model_metrics[model][rst]['precision'] for rst in data_splits.keys()])
        model_averages[model]['recall'] = np.mean([model_metrics[model][rst]['recall'] for rst in data_splits.keys()])
        model_averages[model]['f1'] = np.mean([model_metrics[model][rst]['f1'] for rst in data_splits.keys()])
    
    df_model_metrics = pd.DataFrame.from_dict(model_averages, orient='index')
    print(df_model_metrics)
    maxidx = df_model_metrics['auc'].argmax()
    best_model_name = df_model_metrics['auc'].index.to_list()[maxidx]
    print(f"Best Model (by ROC AUC): {best_model_name}")
    return data_splits, model_res, model_metrics, model_averages, df_model_metrics, best_model_name


def main():
    # SHAP interpretability
    # interpret(MODEL_LIST, data_splits, model_res) # TODO: Format Shapley stuff 

    # Plotting # TODO: Set up properly (possibly in metrics.py as a plotting function)
    # for model in model_names:
    #     plt.bar(['auc', 'acc', 'precision', 'recall', 'f1'], np.array(list(model_metrics[model].values())), color=np.random.rand(3,))
    #     plt.xlabel("Metric")
    #     plt.ylabel("Average (%)")
    #     plt.title(model)
    #     plt.show()

    # Ensemble Model #TODO: What to run it on?
    ensemble = models.ensemble_model([model_info[1][1] for model_info in rare_model_res[rare_best_model_name].items()],
                                     [model_info[1][1] for model_info in clr_model_res[clr_best_model_name].items()])

if __name__ == "__main__":
    # X, y = make_classification(n_samples=20, n_features=5, random_state=42)
    # test_data = np.concatenate((X, y.reshape(-1, 1)), axis = 1)

    rarefaction_data = pd.read_csv("rarefied-feature-table-labeled.csv")
    data_splits, model_res, model_metrics, model_averages, df_model_metrics, best_model_name = pipeline(rarefaction_data.iloc[:,1:].to_numpy())

    for model in MODEL_LIST:
        plt.bar(['auc', 'acc', 'precision', 'recall', 'f1'], np.array(list(model_averages[model].values())), color=np.random.rand(3,))
        plt.xlabel("Metric")
        plt.ylabel("Average (%)")
        plt.title(model)
        plt.show()

    # CLR 
    # X, y = make_classification(n_samples=20, n_features=5, random_state=42)
    # data = np.concatenate((X, y.reshape(-1, 1)), axis = 1)
    # _ = pipeline(data)