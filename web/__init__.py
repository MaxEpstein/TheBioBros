from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/accounts")
def hello_accounts():
    return render_template('accounts.html')

@app.route("/add")
def hello_add():
    return render_template('add-product.html')

@app.route("/edit")
def hello_edit():
    return render_template('edit-product.html')

@app.route("/login")
def hello_login():
    return render_template('login.html')

@app.route("/products")
def hello_products():
    return render_template('products.html')

if __name__ == "__gui__":
    app.run(debug=True)