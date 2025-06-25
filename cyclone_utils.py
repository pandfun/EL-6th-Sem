# import torch
# import torch.nn as nn
# import numpy as np
# import xarray as xr

# class LSTMClassifier(nn.Module):
#     def __init__(self, input_size, hidden_size=64):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         last = out[:, -1, :]
#         return torch.sigmoid(self.fc(last)).squeeze()

# def load_cyclone_model(model_path, input_size):
#     model = LSTMClassifier(input_size)
#     model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
#     model.eval()
#     return model

# def predict_cyclone(X_seq, model):
#     sample_x = torch.from_numpy(X_seq[-1]).unsqueeze(0)
#     with torch.no_grad():
#         prob = model(sample_x).item()
#     if prob > 0.7:
#         return "⚠️ High chance of cyclone imminence!", prob
#     elif prob > 0.3:
#         return "🌥 Moderate chance of cyclone activity.", prob
#     else:
#         return "✅ Low likelihood of cyclone.", prob


import torch
import torch.nn as nn
import numpy as np
import xarray as xr

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return torch.sigmoid(self.fc(last)).squeeze()

def load_cyclone_model(model_path, input_size):
    model = LSTMClassifier(input_size)
    # Use map_location='cpu' if you are not running on a GPU in your Flask app environment
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# Modify this function to accept a single sequence
def predict_cyclone(current_sequence, model):
    """
    Predicts cyclone probability for a single sequence of meteorological data.

    Args:
        current_sequence (np.ndarray): A numpy array of shape (SEQ_LEN, D)
                                       representing the most recent sequence
                                       of data (e.g., 10 time steps).
        model (torch.nn.Module): The loaded PyTorch model.

    Returns:
        tuple: A string message and the predicted probability.
    """
    # Ensure the input is a torch tensor and has the correct shape (batch_size=1)
    sample_x = torch.from_numpy(current_sequence).unsqueeze(0)

    with torch.no_grad():
        prob = model(sample_x).item()

    if prob > 0.7:
        return "⚠️ High chance of cyclone imminence!", prob
    elif prob > 0.3:
        return "🌥 Moderate chance of cyclone activity.", prob
    else:
        return "✅ Low likelihood of cyclone.", prob

# Example usage in your Flask app:
# Assuming you have loaded your model:
# model_path = "cyclone_model.pth"
# input_size = 53248 # This should match the D value from your data processing
# loaded_model = load_cyclone_model(model_path, input_size)

# When you receive new data in your Flask app, you would prepare the
# 'current_sequence' (the last 10 time steps of meteorological data)
# For example, if you have a function to get the latest data:
# latest_data = get_latest_meteorological_data() # This would return a numpy array of shape (10, 53248)

# Then call the predict function with this sequence:
# message, probability = predict_cyclone(latest_data, loaded_model)
# print(message, probability)