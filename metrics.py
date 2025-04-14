# Evaluation Metrics:
    # Area Under the Curve (AUC)
    # Accuracy
    # Precision
    # Recall
    # F1 score.

# Interpretability Metrics:
    # SHAP

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import shap
import numpy as np
import base64
import io
import matplotlib.pyplot as plt


def getMetrics(y_test, x_test, y_pred, model):
    # Returns the ROC AUC, Accuracy, Precision, Recall, 
    # and F1 Score for a model given its prediction and true class information
    y_prob = model.predict_proba(x_test)[:,1]
    areaUnder = roc_auc_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return  areaUnder, accuracy, precision, recall, f1

def interpret(list_of_models, list_of_splits, model_results):
    # From a list of models and train-test splits,
    # generates the average SHAP values for each feature.
    plot_dict = {}
    linear = ["lr", "lr1", "lr2", "lre", "lsv"]
    tree = ["xgb", "lgb", "et", "gb", "dt", "rf"] 
    not_supported = ["ab", "nlsv", "knn", "lda", "gnb", "mlp"]

    for model_name in list_of_models:
        print(model_name)
        shap_explanations = []
        X_test = None

        if model_name in not_supported:
            print(f"{model_name} is not supported")
            plot_dict[model_name] = 'unsupported'
            continue

        for state in list_of_splits.keys():
            # X_train = list_of_splits[state][0]
            # get X_test
            X_test = list_of_splits[state][1]
            model = model_results[model_name][state][1]

            if model_name in linear:
                masker = shap.maskers.Independent(X_test)
                explainer = shap.LinearExplainer(model, masker)
            elif model_name in tree:
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model)

            explanation = explainer(X_test)
            shap_explanations.append(explanation)

        all_values = np.array([exp.values for exp in shap_explanations])
        median_values = np.median(all_values, axis=0)
        feature_names = shap_explanations[0].feature_names
        print(feature_names)

        summary_explanation = shap.Explanation(
            values=median_values,
            base_values=shap_explanations[0].base_values,
            data=X_test,
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
        buff.seek(0)
        plot_base64 = str(base64.b64encode(buff.read()).decode())
        plot_dict[model_name] = plot_base64

    return plot_dict