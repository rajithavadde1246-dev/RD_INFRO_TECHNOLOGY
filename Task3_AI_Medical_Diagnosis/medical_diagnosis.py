import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = pd.read_csv("medical_data.csv")

# Data Analysis
print("First 5 Records:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())

# NumPy usage
symptom_array = np.array(data[["fever", "cough", "headache", "fatigue"]])
print("\nNumPy Array Shape:", symptom_array.shape)

# Features and target
X = data[["fever", "cough", "headache", "fatigue"]]
y = data["disease"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Sample prediction
prediction = model.predict([[1, 1, 1, 1]])

print("\nPredicted Disease:", prediction[0])