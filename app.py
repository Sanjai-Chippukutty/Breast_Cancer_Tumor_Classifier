import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("models/classifier.pkl")

# Page title
st.title("🧬 Breast Cancer Tumor Classifier")
st.markdown("Enter the following 30 features to predict if the tumor is **Benign (0)** or **Malignant (1)**.")

# Feature names
features = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# Input sliders
input_data = []
for feature in features:
    value = st.number_input(f"{feature.title()}", step=0.01)
    input_data.append(value)

# Prediction
if st.button("Predict Tumor Type"):
    input_np = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_np)[0]
    result = "🔵 Benign (0)" if prediction == 0 else "🔴 Malignant (1)"
    st.success(f"Prediction: {result}")
