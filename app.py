from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import torch

from tensorflow.keras.models import load_model
from cyclone_utils import load_cyclone_model, predict_cyclone

app = Flask(__name__)

# Load models once
flood_model = load_model("flood_model.keras")
scaler = joblib.load("scaler.pkl")

last_seq = np.load("last_X_seq.npy")  # Load only once at startup
cyclone_model = load_cyclone_model("cyclone_model.pth", input_size=last_seq.shape[1])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/flood", methods=["GET", "POST"])
def flood():
    prediction = None
    if request.method == "POST":
        try:
            features = [float(request.form[key]) for key in request.form]
            input_df = pd.DataFrame([features], columns=request.form.keys())
            input_scaled = scaler.transform(input_df)
            prob = float(flood_model.predict(input_scaled)[0][0])
            prediction = round(prob, 3)
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template("flood.html", prediction=prediction)

@app.route("/cyclone")
def cyclone():
    message, prob = predict_cyclone(last_seq, cyclone_model)
    return render_template("cyclone.html", message=message, prob=round(prob, 3))

if __name__ == "__main__":
    app.run(debug=True)
