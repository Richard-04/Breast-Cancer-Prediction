import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# -----------------------
# Load model and scaler
# -----------------------
model = keras.models.load_model("breast_cancer_predictor.keras")
scaler = joblib.load("scaler.pkl")  # Use the scaler you fitted during training

# -----------------------
# Page setup
# -----------------------
st.set_page_config(page_title="🎗️ Breast Cancer Predictor", layout="centered")

st.title("🎗️ Breast Cancer Prediction App")
st.markdown("Enter the tumor details to predict if it is likely benign or malignant 📥")

# -----------------------
# CSS styling for sliders
# -----------------------
st.markdown("""
<style>
div[data-baseweb="slider"] input {
    accent-color: purple;
}
div[data-baseweb="slider"] span {
    color: #7e5bef !important;
    font-family: 'Comic Sans MS', cursive, sans-serif;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

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
# Collect user input in two columns
st.subheader("Tumor Features🎗️")
user_data = {}

cols = st.columns(2)  # Create 2 columns

for i, feature in enumerate(feature_names):
    label = feature.replace("_", " ").title()
    col = cols[i % 2]  # alternate between the two columns
    # Use slider in each column
    user_data[feature] = col.slider(label, 0.0, 100.0, 0.0)

# Convert to DataFrame
input_df = pd.DataFrame([user_data])

# -----------------------
# Predict button
# -----------------------
if st.button("PREDICT 🎯"):
    # Scale features
    input_scaled = scaler.transform(input_df)

    # Make prediction
    pred_prob = model.predict(input_scaled)[0][0]
    pred_class = 1 if pred_prob > 0.5 else 0

    if pred_class == 1:
        st.success(f"Malignant tumor predicted. Consult a specialist immediately🏥! ({pred_prob*100:.2f}% probability) ")
    else:
        st.success(f"🎉 Good news! Benign tumor predicted. ({(1-pred_prob)*100:.2f}% probability) ")

