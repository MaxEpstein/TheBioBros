from flask import Flask, render_template, request
from main import pipeline
import pandas as pd

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def run_app():
    if request.method == 'POST':
        rare = request.files['filerare']
        clr = request.files['fileclr']
        model_dict = {
        'dt': 'Decision Tree',
        'gb': 'Gradient Boosting',
        'rf': 'Random Forest',
        'xgb': 'Extreme Gradient Boosting',
        'lgb': 'Light Gradient Boosting',
        'et': 'Extra Trees',
        'ab': 'AdaBoost',
        'lr': 'Logistic Regression',
        'lr1': 'Lasso Regularization',
        'lr2': 'Ridge Regularization',
        'lre': 'Elastic Net Regularization',
        'lsv': 'Linear Support Vector',
        'nlsv': 'Non-Linear Support Vector',
        'knn': 'k-Nearest Neighbor',
        'lda': 'Linear Discriminant',
        'gnb': 'Gaussian Naive-Bayes',
        'mlp': 'Multi-Layer Perception'
        }
        if rare:
            df = pd.read_csv(rare)
            _, _, _, _, _, rare_model_name, rare_plot, rare_list = pipeline(df.iloc[:,1:].to_numpy(), feature_names=df.iloc[:,1:-1].columns.tolist())
        if clr:
            df = pd.read_csv(clr)
            _, _, _, _, _, clr_model_name, clr_plot, clr_list = pipeline(df.iloc[:,1:].to_numpy(), feature_names=df.iloc[:,1:-1].columns.tolist())
        return render_template('submitted.html', rare_model_name=model_dict[rare_model_name], clr_model_name=model_dict[clr_model_name],
                               rare_plot=rare_plot, rare_list=rare_list, rare_model=rare_model_name,
                               clr_plot=clr_plot, clr_list=clr_list, clr_model=clr_model_name, model_dict=model_dict)
    if request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')

if __name__ == "__gui__":
    app.run(debug=True)