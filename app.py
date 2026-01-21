import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------
# Load model and scaler
# -----------------------
model = joblib.load("model/breast_cancer_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# -----------------------
# Page setup
# -----------------------
st.set_page_config(page_title="🎗️ Breast Cancer Predictor", layout="centered")

st.title("🎗️ Breast Cancer Prediction App")
st.markdown(
    "Enter the tumor details below to predict whether the tumor is **Benign** or **Malignant**."
)

# -----------------------
# Feature list (30 features)
# -----------------------
feature_names = [
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness",
    "mean_compactness", "mean_concavity", "mean_concave_points", "mean_symmetry", "mean_fractal_dimension",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# -----------------------
# Collect user input
# -----------------------
st.subheader("Tumor Features")
user_data = {}

cols = st.columns(2)

for i, feature in enumerate(feature_names):
    label = feature.replace("_", " ").title()
    col = cols[i % 2]
    user_data[feature] = col.slider(
        label=label,
        min_value=0.0,
        max_value=100.0,
        value=0.0
    )

# Convert to DataFrame
input_df = pd.DataFrame([user_data])

# -----------------------
# Prediction
# -----------------------
if st.button("PREDICT 🎯"):
    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("❌ **Malignant Tumor Detected** — Please consult a medical professional immediately.")
    else:
        st.success("✅ **Benign Tumor Detected** — No immediate cause for concern.")
