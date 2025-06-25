# 📦 Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# 📁 Load data
data = pd.read_csv('data/flood.csv')

# ✅ Outlier Capping Function
def cap_outliers(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    data[col] = data[col].clip(lower, upper)

# 🔍 Cap outliers for all numeric features
numerical_features = data.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_features:
    cap_outliers(data, col)

# 🎯 Convert FloodProbability to binary class for classification
data['FloodProbability'] = data['FloodProbability'].apply(lambda x: 1 if x >= 0.5 else 0)

# 🔧 Splitting Features and Target
X = data.drop(columns=['FloodProbability'])
y = data['FloodProbability']

# 📊 Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧪 Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🧠 Neural Network Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # sigmoid for probability output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ⏱️ Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 🚀 Model Training
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=256, validation_split=0.2, callbacks=[early_stopping])

# 📉 Evaluate Model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Model Loss: {loss:.4f}")
print(f"Model Accuracy: {accuracy:.4f}")

# 📈 Predict probabilities
y_probs = model.predict(X_test_scaled)
y_pred = (y_probs > 0.5).astype(int)

# 📊 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Flood", "Flood"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# 📄 Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 🌍 Predict flood probability for new input
def predict_flood_probability(input_data: dict):
    """
    Predict flood probability for a single input using the trained model.

    Args:
        input_data (dict): Feature dictionary with same keys as training data.

    Returns:
        float: Predicted flood probability (0.0 to 1.0)
    """
    try:
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        probability = float(model.predict(input_scaled)[0][0])
        return round(probability, 4)
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return -1

# 🧪 Example input
example_input = {
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

# 🔍 Predict
prob = predict_flood_probability(example_input)
if prob != -1:
    print(f"🌊 Predicted Flood Probability: {prob:.4f}")


model.save("data/flood_model.keras")

import joblib
joblib.dump(scaler, "data/scaler.pkl")

