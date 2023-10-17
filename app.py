import pandas as pd
from flask import Flask, requests, jsonify, render_template
import pickle
# Create the flask app
app = Flask(__name__)

# Cargar el modelo Pickle
model = pickle.load(open("model.pkl", "rb"))

# Definir las rutas
@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods= ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "La especie es: {}".format(prediction))

if __name__ == "__main__":
    app.run(debug = True)
