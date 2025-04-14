import numpy as np
import preprocess
import modeling as models
from metrics import getMetrics
import pandas as pd
from metrics import interpret
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from testKvalue import run_k_test
import base64
import io

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_LIST = ["dt", "rf", "gb", "xgb", "lgb", "et","ab", "lr", "lr1", "lr2", "lre", "lsv", "nlsv", "knn", "lda", "gnb", "mlp"]

def pipeline(data, selection=None, extraction=None, k=20, create_metric_plots=True, create_interpret_plot=False, run_test=False):
    np.random.seed(42)
    states = np.random.randint(low = 0, high = 1000000, size=(100,))
    data_splits = {}
    metric_plot_dict = {}
    interpret_plot_dict = None
    for rst in states:
        data_splits[rst] = preprocess.preprocess(data, rst, selection=selection, extraction=extraction, k=k) # TODO: figure out about k
    
    model_res = {}
    for model in MODEL_LIST:
        model_res[model] = {}

    for i, state in enumerate(data_splits.keys()):    
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
        print(f"Finished training models on round: {i}")

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
    auc_sort_dec = df_model_metrics['auc'].argsort()[::-1]
    model_auc_dec = df_model_metrics['auc'].index.to_list()[auc_sort_dec]
    best_model_name = df_model_metrics['auc'].index.to_list()[auc_sort_dec[0]]
    print(f"Best Model (by ROC AUC): {best_model_name}")
    if create_metric_plots:
        for model in model_auc_dec:
            plot_b64 = createGraph(model, model_averages) # Store as base64 so we can later send to front end
            metric_plot_dict[model] = plot_b64

    if create_interpret_plot:
        interpret_plot_dict = interpret(model_auc_dec, data_splits, model_res) # Store as base64 so we can later send to front end

    if run_test:
        k_vals = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
        results = run_k_test(data, best_model_name, states=states, k_vals=k_vals)

    return data_splits, model_res, model_metrics, model_averages, df_model_metrics, best_model_name, metric_plot_dict, interpret_plot_dict, model_auc_dec

def createGraph(model_name, metrics):
    buff = io.BytesIO()
    plt.bar(['auc', 'acc', 'precision', 'recall', 'f1'], np.array(list(metrics[model_name].values())), color=np.random.rand(3,))
    plt.xlabel("Metric")
    plt.ylabel("Average (%)")
    plt.title(f'{model_name.capitalize()} Average Metrics')
    plt.savefig(buff, format='jpg')
    buff.seek(0)
    plot_base64 = str(base64.b64encode(buff.read()).decode())
    return plot_base64

if __name__ == "__main__":
    # Rarefaction
    rarefaction_data = pd.read_csv("rarefied-feature-table-labeled.csv")
    rare_data_splits, rare_model_res, rare_model_metrics, rare_model_averages,rare_df_model_metrics, rare_best_model_name, rare_metric_plot_dict, rare_interpret_plot_dict, rare_model_auc_dec = pipeline(rarefaction_data.iloc[:,1:].to_numpy())

    # CLR 
    clr_data = pd.read_csv("reduced-clr-feature-table-labeled.csv")
    clr_data_splits, clr_model_res, clr_model_metrics, clr_model_averages, clr_df_model_metrics, clr_best_model_name, clr_metric_plot_dict, clr_interpret_plot_dict, clr_model_auc_dec = pipeline(clr_data.iloc[:,1:].to_numpy())

    # ensemble = models.ensemble_model([model_info[1][1] for model_info in rare_model_res[rare_best_model_name].items()],
    #                                  [model_info[1][1] for model_info in clr_model_res[clr_best_model_name].items()])
