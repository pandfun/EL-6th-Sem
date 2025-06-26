import pandas as pd
import numpy as np
import requests
import torch
import torch.nn as nn
from datetime import datetime, timedelta

# Define the TinyLSTM model
class TinyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 8, batch_first=True)
        self.fc   = nn.Linear(8, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(self.lstm(x)[0][:, -1]))

# Fetch sequence of recent magnitudes for prediction
def get_latest_eq_sequence(seq_len=5):
    end = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=seq_len)
    url = (
        "https://earthquake.usgs.gov/fdsnws/event/1/query?"
        f"format=geojson&starttime={start.isoformat()}&minmagnitude=0"
    )
    try:
        resp = requests.get(url, timeout=10).json()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

    mags = []
    for feat in resp.get('features', []):
        t = datetime.utcfromtimestamp(feat['properties']['time']/1000)
        t = t.replace(minute=0, second=0, microsecond=0)
        mags.append((t, feat['properties']['mag'] or 0))

    df = pd.DataFrame(mags).groupby(0)[1].max().reindex(
        pd.date_range(start, end - timedelta(hours=1), freq='h'),
        fill_value=0
    ).reset_index(drop=True)

    if len(df) < seq_len:
        return None

    arr = df.values.astype('float32')[-seq_len:]
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(-1)

# Device and model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_eq = TinyLSTM().to(device)
model_eq.eval()
# Optionally: model_eq.load_state_dict(torch.load('path/to/model.pth'))

# Run prediction using the model
def predict_earthquake():
    seq = get_latest_eq_sequence()
    if seq is None:
        return None, "Insufficient data from USGS API."
    seq = seq.to(device)
    prob = model_eq(seq).item()
    return round(prob, 3), "⚠️ Likely" if prob > 0.99 else "✅ Unlikely"
