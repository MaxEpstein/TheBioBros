import numpy as np
import preprocess
import modeling as models
from metrics import get_metrics
import pandas as pd
from metrics import interpret
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from testKvalue import run_k_test, k_test_plots
import base64
import io

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_LIST = ["dt", "rf", "gb", "xgb", "lgb", "et","ab", "lr",
              "lr1", "lr2", "lre", "lsv", "nlsv", "knn", "lda", "gnb", "mlp"]

def pipeline(data, seed=42, selection=None, extraction=None, k=100, num_repeats=100, feature_names=None, create_metric_plots=True, create_interpret_plot=False, run_test=False):
    '''Runs our ML pipeline on the passed in dataset.
    Args:
        data: the dataset
        seed: select random seed to do data splits
        selection: toggling the feature selection algorithm used
        extraction: toggling the feature engineering/extraction algorithm used
        k: the number of features to use for the feature engineering method
        num_repeats: the number of random fits for our models to average over (default: 100)
        feature_names: the names of the columns (i.e. features) to be used for SHAP plots (if enabled)
        create_metric_plots: generates base64 metric plots and ROC curve plots
        create_interpret_plot: generates SHAP plots for models
        run_test: runs feature engineering k value tests for k values 1 - 200 in increments of 10 (WARNING: takes long time to run)

    Returns:
        data_splits: the dataset splits into test and train. Done (num_repeats) times
        
        model_res: contains predicted labels on x_train and fitted models
        
        model_metrics: contains metrics for each individual model
        
        df_model_averages: dataframe of model averages per model type
        
        best_model_name: name of best model
        
        plot_dict: contains base64 versions of all generated plots toggled
        
        model_auc_dec: model names in descending order of highest auc.
    '''
    np.random.seed(seed)
    # Generate random states to do data splits
    states = np.random.randint(low = 0, high = 1000000, size=(num_repeats,))
    data_splits = {}
    plot_dict = {}
    for rst in states:
        data_splits[rst] = preprocess.preprocess(data, rst, selection=selection, extraction=extraction, k=k)
    
    model_res = {}
    for model in MODEL_LIST:
        model_res[model] = {}

    # Fit Data Splits on All 17 Models
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

    # Calculate the metrics for all models
    for rst in data_splits.keys():
        for model in MODEL_LIST:
            model_metrics[model][rst] = {}
            metric = get_metrics(data_splits[rst][3], data_splits[rst][1], model_res[model][rst][0], model_res[model][rst][1])
            model_metrics[model][rst]['auc'] = metric[0]
            model_metrics[model][rst]['acc'] = metric[1]
            model_metrics[model][rst]['precision'] = metric[2]
            model_metrics[model][rst]['recall'] = metric[3]
            model_metrics[model][rst]['f1'] = metric[4]
            model_metrics[model][rst]['fpr'] = metric[5]
            model_metrics[model][rst]['tpr'] = metric[6]
    
    # Average the metrics over all runs
    model_averages = {}
    for model in MODEL_LIST:
        model_averages[model] = {}
        model_averages[model]['auc'] = np.mean([model_metrics[model][rst]['auc'] for rst in data_splits.keys()])
        model_averages[model]['acc'] = np.mean([model_metrics[model][rst]['acc'] for rst in data_splits.keys()])
        model_averages[model]['precision'] = np.mean([model_metrics[model][rst]['precision'] for rst in data_splits.keys()])
        model_averages[model]['recall'] = np.mean([model_metrics[model][rst]['recall'] for rst in data_splits.keys()])
        model_averages[model]['f1'] = np.mean([model_metrics[model][rst]['f1'] for rst in data_splits.keys()])
    
    # Find best model by AUC 
    df_model_averages = pd.DataFrame.from_dict(model_averages, orient='index')
    print(df_model_averages)
    auc_sort_dec = df_model_averages['auc'].to_numpy().argsort()[::-1]
    model_auc_dec = df_model_averages.iloc[auc_sort_dec].index.to_list()
    best_model_name = model_auc_dec[0]
    print(f"Best Model (by ROC AUC): {best_model_name}")
    for model in model_auc_dec:
        plot_dict[model] = {}

    # Create Metric Plots (if selected)
    if create_metric_plots:
        for model in model_auc_dec:
            plot_b64 = create_graph(model, model_averages) # Store as base64 so we can later send to front end
            plot_dict[model]['metrics'] = plot_b64

        roc_plot_dict = plot_avg_roc_base64(model_metrics, model_auc_dec)
        for model in model_auc_dec:
            plot_dict[model]['roc_plot'] = roc_plot_dict[model]

    # Create Interpretability Plots (if selected)
    if create_interpret_plot:
        interpret(model_auc_dec, data_splits, model_res, plot_dict, feature_names) # Store as base64 so we can later send to front end

    # Run K Value Test (if selected)
    if run_test:
        k_vals = [1, 10]
        k_test_results = run_k_test(data, best_model_name, states=states, k_vals=k_vals)
        plot_dict['k_test'] = k_test_plots(k_test_results)

    return data_splits, model_res, model_metrics, df_model_averages, best_model_name, plot_dict, model_auc_dec

def average_roc(model_metrics, model_name, num_points=100):
    "Get average roc based on model metrics"
    mean_fpr = np.linspace(0, 1, num_points)
    tprs = []

    for rst in model_metrics[model_name]:
        fpr = model_metrics[model_name][rst]['fpr']
        tpr = model_metrics[model_name][rst]['tpr']
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    return mean_fpr, mean_tpr


def plot_avg_roc_base64(model_metrics, model_list):
    "Generates ROC Curve and Converts to Base64"
    plot_dict = {}
    for model in model_list:
        mean_fpr, mean_tpr = average_roc(model_metrics, model)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(mean_fpr, mean_tpr, label=f"{model.upper()} (mean AUC = {np.mean([model_metrics[model][rst]['auc'] for rst in model_metrics[model]]):.2f})")
        ax.plot([0, 1], [0, 1], 'k--', label='Random Chance')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"Mean ROC Curve ({model.upper()})")
        ax.legend(loc="lower right")
        ax.grid()

        # Encode plot to base64
        buffer = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buffer, format="png")
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close(fig)

        encoded = base64.b64encode(image_png).decode("utf-8")
        plot_dict[model] = encoded
    return plot_dict


def create_graph(model_name, metrics):
    "Generates metric plot for a model"
    buff = io.BytesIO()
    plt.bar(['auc', 'acc', 'precision', 'recall', 'f1'], np.array(list(metrics[model_name].values())), color=np.random.rand(3,))
    plt.xlabel("Metric")
    plt.ylabel("Average (%)")
    plt.title(f'{model_name.capitalize()} Average Metrics')
    plt.savefig(buff, format='jpg')
    plt.close()
    buff.seek(0)
    plot_base64 = str(base64.b64encode(buff.read()).decode())
    return plot_base64

if __name__ == "__main__":
    # Rarefaction
    rarefaction_data = pd.read_csv("rarefied-feature-table-labeled.csv")
    rare_data_splits, rare_model_res, rare_model_metrics, rare_df_model_metrics, rare_best_model_name, rare_plot_dict, rare_model_auc_dec = pipeline(rarefaction_data.iloc[:,1:].to_numpy(), feature_names=rarefaction_data.iloc[:,1:-1].columns.tolist())

    # CLR 
    clr_data = pd.read_csv("reduced-clr-feature-table-labeled.csv")
    clr_data_splits, clr_model_res, clr_model_metrics, clr_df_model_metrics, clr_best_model_name, clr_plot_dict, clr_model_auc_dec = pipeline(clr_data.iloc[:,1:].to_numpy(), feature_names=clr_data.iloc[:,1:-1].columns.tolist())

    ## If ensemble model wanted, this code will return the ensemble model based on the above results of rarefaction and clr best models:
    # ensemble = models.ensemble_model([model_info[1][1] for model_info in rare_model_res[rare_best_model_name].items()],
    #                                  [model_info[1][1] for model_info in clr_model_res[clr_best_model_name].items()])
