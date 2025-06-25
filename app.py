from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import torch
import requests
import time
import json

from tensorflow.keras.models import load_model
from cyclone_utils import load_cyclone_model, predict_cyclone

app = Flask(__name__)

# === Load Prediction Models ===
flood_model = load_model("data/flood_model.keras")
scaler = joblib.load("data/scaler.pkl")
last_seq = np.load("data/last_X_seq.npy")
cyclone_model = load_cyclone_model("data/cyclone_model.pth", input_size=last_seq.shape[1])

# === News & Chatbot Config ===
NEWS_API_KEY = "KEY_HERE"
NEWS_BASE_URL = "https://newsdata.io/api/1/news"
OLLAMA_API_URL = "http://localhost:11434/api/generate"

DEFAULT_QUERY = "earthquake OR flood OR wildfire OR hurricane"
SDG11_QUERY = "urban resilience OR sustainable cities OR SDG11 OR climate adaptation OR emergency preparedness"

# === Routes ===
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/flood', methods=['GET', 'POST'])
def flood():
    prediction = None
    field_values = {
        "MonsoonIntensity": 2,
        "TopographyDrainage": 8,
        "RiverManagement": 9,
        "Deforestation": 1,
        "Urbanization": 2,
        "ClimateChange": 2,
        "DamsQuality": 9,
        "Siltation": 1,
        "AgriculturalPractices": 8,
        "Encroachments": 1,
        "IneffectiveDisasterPreparedness": 1,
        "DrainageSystems": 9,
        "CoastalVulnerability": 1,
        "Landslides": 1,
        "Watersheds": 9,
        "DeterioratingInfrastructure": 1,
        "PopulationScore": 2,
        "WetlandLoss": 1,
        "InadequatePlanning": 1,
        "PoliticalFactors": 1
    }

    if request.method == "POST":
        try:
            for key in field_values:
                # Overwrite default with submitted value
                field_values[key] = float(request.form.get(key, field_values[key]))

            input_df = pd.DataFrame([field_values])
            input_scaled = scaler.transform(input_df)
            prob = float(flood_model.predict(input_scaled)[0][0])
            prediction = round(prob, 3)
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("flood.html", prediction=prediction, field_values=field_values)




@app.route('/cyclone')
def cyclone():
    message, prob = predict_cyclone(last_seq, cyclone_model)
    return render_template("cyclone.html", message=message, prob=round(prob, 3))

@app.route('/get-tips', methods=['POST'])
def get_tips():
    disaster = request.json.get('disaster')
    if not disaster:
        return jsonify({"error": "No disaster name provided"})
    tips = get_prevention_tips(disaster)
    return jsonify({"tips": tips, "disaster": disaster})

@app.route('/fetch-alerts', methods=['GET'])
def fetch_alerts():
    query = request.args.get('q', default=DEFAULT_QUERY, type=str)
    if SDG11_QUERY in query:
        return jsonify(load_mock_data())
    try:
        response = requests.get(NEWS_BASE_URL, params={
            'apiKey': NEWS_API_KEY,
            'q': query,
            'language': 'en'
        })

        data = response.json()
        results = data.get('results', [])

        if isinstance(results, list) and results:
            filtered = filter_results(results, query)
            return jsonify({"results": filtered[:6]})
        else:
            return jsonify({"results": [], "error": "No relevant news found."})

    except Exception as e:
        return jsonify({"results": [], "error": str(e)})

# === Helper Functions ===
def get_prevention_tips(disaster):
    payload = {
        "model": "llama3.2:latest",
        "prompt": (
            f"List exactly 5 short and clear prevention measures for {disaster} as bullet points only.\n"
            "Do NOT explain anything, do NOT add extra text before or after, and use a new line for each point.\n"
            "Format:\n"
            "Tip 1\n"
            "Tip 2\n"
            "Tip 3\n"
            "Tip 4\n"
            "Tip 5"
        ),
        "stream": False,
        "temperature": 0.1,
        "max_tokens": 150
    }
    try:
        start_time = time.time()
        response = requests.post(OLLAMA_API_URL, json=payload)
        end_time = time.time()
        print(f"⏱️ Response time for '{disaster}': {end_time - start_time:.2f}s")
        if response.status_code == 200:
            raw_response = response.json().get("response", "")
            cleaned_tips = clean_tips(raw_response)
            return '\n'.join(cleaned_tips) if cleaned_tips else "No tips available."
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Connection error: {e}"

def clean_tips(raw_text):
    lines = raw_text.strip().split('\n')
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(('-', '*', '•')) or len(cleaned) < 5:
            cleaned.append(line.lstrip('-*• ').strip())
        if len(cleaned) == 5:
            break
    return [f"- {tip}" for tip in cleaned if tip]

def filter_results(results, query):
    keywords = [kw.strip().lower() for kw in query.split("OR")]
    filtered = []
    for article in results:
        content = f"{article.get('title', '')} {article.get('description', '')}".lower()
        if any(keyword in content for keyword in keywords):
            filtered.append(article)
    return filtered

def load_mock_data():
    return {
        "results": [
            {
                "title": "Urban Resilience: Building Sustainable Cities for Climate Adaptation",
                "description": "New initiatives focus on creating climate-resilient urban infrastructure to protect communities from disasters.",
                "link": "https://example.com/urban-resilience",
                "image_url": "https://via.placeholder.com/400x200?text=Urban+Resilience",
                "source_id": "Sustainability News",
                "pubDate": "2024-12-15T10:00:00Z"
            },
            {
                "title": "SDG 11: Making Cities Inclusive, Safe, and Sustainable",
                "description": "Progress report on Sustainable Development Goal 11 highlights achievements in urban planning and disaster preparedness.",
                "link": "https://example.com/sdg11-progress",
                "image_url": "https://via.placeholder.com/400x200?text=SDG+11",
                "source_id": "UN News",
                "pubDate": "2024-12-14T15:30:00Z"
            },
            {
                "title": "Emergency Preparedness in Smart Cities",
                "description": "Technology integration in urban planning enhances emergency response and disaster management capabilities.",
                "link": "https://example.com/smart-cities-emergency",
                "image_url": "https://via.placeholder.com/400x200?text=Smart+Cities",
                "source_id": "Tech Today",
                "pubDate": "2024-12-13T08:45:00Z"
            }
        ]
    }


if __name__ == '__main__':
    app.run(debug=True, port=5050)
