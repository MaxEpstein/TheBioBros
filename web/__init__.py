from flask import Flask, render_template, request
import csv, io
from main import pipeline
import pandas as pd
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        rare = request.files['filerare']
        clr = request.files['fileclr']
        if rare:
            df = pd.read_csv(rare)
            print(df.head())
            df = df.drop(columns = ['Diagnosis'])
            _, _, _, _, _, rare_model_name, rare_metrics, rare_interpret, rare_list = pipeline(df.iloc[:,1:].to_numpy())
            print(rare_model_name)
            rare_image_metrics = rare_metrics
        if clr:
            df = pd.read_csv(clr)
            print(df.head())
            df = df.drop(columns = ['Diagnosis'])
            _, _, _, _, _, clr_model_name, clr_metrics, clr_interpret, clr_list = pipeline(df.iloc[:,1:].to_numpy())
            print(clr_model_name)
        return render_template('submitted.html', rare_model_name=rare_model_name, clr_model_name=clr_model_name,
                               rare_metrics=rare_metrics, rare_interpret=rare_interpret, rare_list=rare_list, 
                               clr_metrics=clr_metrics, clr_interpret=clr_interpret, clr_list=clr_list)
    if request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')

if __name__ == "__gui__":
    app.run(debug=True)