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
</style>
</head>
<body>

<h1>GutCheck</h1>
<h3>Microbiome ML Pipeline for Analysis and Interpretation</h3>
<p>This is a paragraph.</p>
<div>This is a div.</div>

</body>
</html>'''