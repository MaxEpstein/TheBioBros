# Evaluation Metrics:
    # Area Under the Curve (AUC)
    # Accuracy
    # Precision
    # Recall
    # F1 score.

# Interpretability Metrics:
    # SHAP

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
import shap
import numpy as np
import base64
import io
import matplotlib.pyplot as plt


def get_metrics(y_test, x_test, y_pred, model):
    '''Returns the ROC AUC, Accuracy, Precision, Recall, and F1 Score for a model
       given its prediction and true class information
    '''
    # Returns the ROC AUC, Accuracy, Precision, Recall, 
    # and F1 Score for a model given its prediction and true class information
    y_prob = model.predict_proba(x_test)[:,1]
    auc = roc_auc_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    return  auc, accuracy, precision, recall, f1, fpr, tpr

def interpret(list_of_models, list_of_splits, model_results, dictionary, feature_names=None):
    # From a list of models and train-test splits,
    # generates the average SHAP values for each feature.
    '''From a list of models and train-test splits,
       generates the average SHAP values for each feature and plots.
    Args:
        list_of_models: model names to generate values for
        list_of_splits: all train-test splits
        model_results: predictions and fitted models for all splits
        dictionary: the dictionary we want to pass our results to
    '''

    linear = ["lr", "lr1", "lr2", "lre", "lsv"]
    tree = ["xgb", "lgb", "et", "gb", "dt", "rf"] 
    not_supported = ["ab", "nlsv", "knn", "lda", "gnb", "mlp"]

    for model_name in list_of_models:
        print(model_name)
        shap_explanations = []
        x_test = None

        if model_name in not_supported:
            print(f"{model_name} is not supported")
            dictionary[model_name]['interpret'] = 'unsupported'
            continue

        for state in list_of_splits.keys():
            # X_train = list_of_splits[state][0]
            # get X_test
            x_test = list_of_splits[state][1]
            model = model_results[model_name][state][1]

            if model_name in linear:
                masker = shap.maskers.Independent(x_test)
                explainer = shap.LinearExplainer(model, masker)
            elif model_name in tree:
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model)

            explanation = explainer(x_test)
            shap_explanations.append(explanation)

        all_values = np.array([exp.values for exp in shap_explanations])
        median_values = np.median(all_values, axis=0)

        if feature_names is None:
            feature_names = shap_explanations[0].feature_names

        summary_explanation = shap.Explanation(
            values=median_values,
            base_values=shap_explanations[0].base_values,
            data=x_test,
            feature_names=feature_names
        )
        buff = io.BytesIO()
        shap.summary_plot(summary_explanation, 
                          plot_type="dot", 
                          max_display=10, 
                          color_bar_label='Microbiome Test',
                          show=False
        )
        plt.savefig(buff, format='jpg')
        plt.close()
        buff.seek(0)
        plot_base64 = str(base64.b64encode(buff.read()).decode())
        dictionary[model_name]['interpret'] = plot_base64