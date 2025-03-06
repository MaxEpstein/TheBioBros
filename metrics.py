# Evaluation Metrics:
    # Area Under the Curve (AUC)
    # Accuracy
    # Precision
    # Recall
    # F1 score.

# Interpretability Metrics:
    # SHAP

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
def getMetrics(y_test, x_test, y_pred, model):
    y_prob = model.predict_proba(x_test)[:,1]
    areaUnder = roc_auc_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return  areaUnder, accuracy, precision, recall, f1
