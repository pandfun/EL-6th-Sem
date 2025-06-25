from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and scaler
model = load_model("flood_model.keras")
scaler = joblib.load("scaler.pkl")

# Features used by the model
FEATURES = [
    "MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Deforestation",
    "Urbanization", "ClimateChange", "DamsQuality", "Siltation",
    "AgriculturalPractices", "Encroachments", "IneffectiveDisasterPreparedness",
    "DrainageSystems", "CoastalVulnerability", "Landslides", "Watersheds",
    "DeterioratingInfrastructure", "PopulationScore", "WetlandLoss",
    "InadequatePlanning", "PoliticalFactors"
]

# Default values (preloaded into form)
DEFAULT_INPUT = {
    "MonsoonIntensity": 7, "TopographyDrainage": 5, "RiverManagement": 6,
    "Deforestation": 7, "Urbanization": 6, "ClimateChange": 7,
    "DamsQuality": 5, "Siltation": 6, "AgriculturalPractices": 6,
    "Encroachments": 5, "IneffectiveDisasterPreparedness": 6,
    "DrainageSystems": 6, "CoastalVulnerability": 5, "Landslides": 5,
    "Watersheds": 6, "DeterioratingInfrastructure": 6, "PopulationScore": 7,
    "WetlandLoss": 6, "InadequatePlanning": 6, "PoliticalFactors": 5
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            user_input = {feature: float(request.form[feature]) for feature in FEATURES}
            input_df = pd.DataFrame([user_input])
            input_scaled = scaler.transform(input_df)
            prediction = float(model.predict(input_scaled)[0][0])
            return render_template('index.html', prediction=round(prediction, 4), default_values=user_input)
        except Exception as e:
            return render_template('index.html', error=str(e), default_values=DEFAULT_INPUT)
    return render_template('index.html', default_values=DEFAULT_INPUT)

if __name__ == '__main__':
    app.run(debug=True)
