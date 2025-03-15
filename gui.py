from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return '''
<html>
<head>
<title>GutCheck Beta</title>
<style>
h1 {text-align: center;}
h3 {text-align: center;}
div {text-align: center;}
</style>
</head>
<body>

<h1>GutCheck</h1>
<h3>Microbiome ML Pipeline for Analysis and Interpretation</h3>
<div><button type="button">Upload Dataset</button></div>
<h3>Preprocessing</h3>
<div><table align="center" style="margin: 0px auto;" style="background-color: #808080;">
  <tr>
    <th>Preprocessing Step</th>
    <th>Progress</th>
  </tr>
  <tr>
    <td>Train-Test Split</td>
    <td>✔️ Success</td>
  </tr>
  <tr>
    <td>Feature Selection</td>
    <td>In Progress...</td>
  </tr>
</table></div>
<h3>Model Training</h3>
<h3>Model Evaluation</h3>
<h3>Model Interpretation</h3>

</body>
</html>'''