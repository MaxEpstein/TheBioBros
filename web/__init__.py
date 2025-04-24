from flask import Flask, render_template, request
from main import pipeline
import pandas as pd

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def run_app():
    if request.method == 'POST': # Logic when uploading files
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
        'mlp': 'Multi-Layer Perceptron'
        } # Used to convert codes into understandable names
        if rare:
            df = pd.read_csv(rare) # Read and train on rarefaction dataset
            _, _, _, _, best_rare_model, rare_plot, rare_list = pipeline(df.iloc[:,1:].to_numpy(), feature_names=df.iloc[:,1:-1].columns.tolist(), create_interpret_plot=True, num_repeats=1, run_test=True)
        if clr:
            df = pd.read_csv(clr) # Read and train on CLR dataset 
            _, _, _, _, best_clr_model, clr_plot, clr_list = pipeline(df.iloc[:,1:].to_numpy(), feature_names=df.iloc[:,1:-1].columns.tolist(), create_interpret_plot=True, num_repeats=1, run_test=True)
            # Displays the page after submitting data, passing variables to the HTML
        return render_template('submitted.html', rare_model_name=model_dict[best_rare_model], clr_model_name=model_dict[best_clr_model],
                               rare_plot=rare_plot, rare_list=rare_list, best_rare_model=best_rare_model,
                               clr_plot=clr_plot, clr_list=clr_list, best_clr_model=best_clr_model, model_dict=model_dict)
    if request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html') # Displays the first page

if __name__ == "__gui__":
    app.run(debug=False)