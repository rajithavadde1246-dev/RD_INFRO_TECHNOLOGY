import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = pd.read_csv("medical_data.csv")

X = data[["fever", "cough", "headache", "fatigue"]]
y = data["disease"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

st.title("AI Medical Diagnosis System")

fever = st.selectbox("Fever", [0, 1])
cough = st.selectbox("Cough", [0, 1])
headache = st.selectbox("Headache", [0, 1])
fatigue = st.selectbox("Fatigue", [0, 1])

if st.button("Predict Disease"):
    prediction = model.predict([[fever, cough, headache, fatigue]])
    st.success(f"Predicted Disease: {prediction[0]}")