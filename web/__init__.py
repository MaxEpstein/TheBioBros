from flask import Flask, render_template, request
import csv, io
from main import pipeline
import pandas as pd

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        file = request.files['filename']
        if file:
            df = pd.read_csv(file)
            print(df.head())
            df = df.drop(columns = ['Diagnosis'])
            _, _, _, _, _, model_name = pipeline(df.iloc[:,1:].to_numpy())
            print(model_name)
        return render_template('index.html')
    if request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')

if __name__ == "__gui__":
    app.run(debug=True)