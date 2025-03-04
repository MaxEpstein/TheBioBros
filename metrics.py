# Evaluation Metrics:
    # Area Under the Curve (AUC)
    # Accuracy
    # Precision
    # Recall
    # F1 score.

# Interpretability Metrics:
    # SHAP

from sklearn.metrics import auc, accuracy_score, precision_score, recall_score, f1_score
def getMetrics(x_val, y_val):
    #areaUnder = auc(x_val, y_val)
    accuracy = accuracy_score(x_val, y_val)
    precision = precision_score(x_val, y_val)
    recall = recall_score(x_val, y_val)
    f1 = f1_score(x_val, y_val)
    return  accuracy, precision, recall, f1
