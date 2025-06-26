import pandas as pd
import numpy as np
import requests
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
import io

# Define the LSTM model class
class TinyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 8, batch_first=True)
        self.fc   = nn.Linear(8, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out[:, -1])).squeeze()

# Define the function to get the latest water level sequence
def get_latest_sequence(seq_len=5, station="9410230"):
    now   = datetime.utcnow()
    start = (now - timedelta(hours=seq_len+1)).strftime("%Y%m%d %H:%M")
    end   = now.strftime("%Y%m%d %H:%M")

    base_url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    params = {
        "begin_date": start,
        "end_date"  : end,
        "station"   : station,
        "product"   : "water_level",
        "datum"     : "MLLW",
        "units"     : "metric",
        "time_zone" : "GMT",
        "format"    : "csv",
    }

    # use params to let requests handle encoding
    resp = requests.get(base_url, params=params)
    df_rt = pd.read_csv(io.StringIO(resp.text), parse_dates=['Date Time'])
    df_rt.columns = df_rt.columns.str.strip()

    if 'Water Level' not in df_rt.columns:
        print("Error: 'Water Level' missing in API response.", df_rt.columns.tolist())
        return None

    # aggregate to hourly
    df_rt['time'] = df_rt['Date Time'].dt.floor('h')
    hourly_rt = (
        df_rt
        .groupby('time')['Water Level']
        .mean()
        .reset_index()
        .rename(columns={'Water Level':'WL'})
    )

    if len(hourly_rt) < seq_len:
        print("Error: insufficient data points. Got", len(hourly_rt))
        return None

    seq = hourly_rt['WL'].values[-seq_len:].astype(np.float32)
    return torch.from_numpy(seq).unsqueeze(0).unsqueeze(-1)

# Define the prediction function
def predict_tsunami(model, seq):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    seq = seq.to(device)
    model.eval()
    with torch.no_grad():
        prob = model(seq).item()
    return prob

if __name__ == '__main__':
    # Example usage (for testing the .py file)
    # This part will not run when imported into Flask
    model = TinyLSTM()
    # In a real application, you would load your trained model state here
    # model.load_state_dict(torch.load('your_model.pth'))

    latest_sequence = get_latest_sequence()
    if latest_sequence is not None:
        probability = predict_tsunami(model, latest_sequence)
        print(f"Predicted probability of tsunami: {probability:.4f}")
        print("ALERT!" if probability > 0.99 else "Safe")