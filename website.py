import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

base_path = os.path.dirname(os.path.abspath(__file__))  # Absolute path of current script

try:
    model = joblib.load(os.path.join(base_path, "yield_model.pkl"))
    scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
    X_columns = joblib.load(os.path.join(base_path, "X_columns.pkl"))
    global_area_lambda = joblib.load(os.path.join(base_path, "global_area_lambda.pkl"))
    global_shift = joblib.load(os.path.join(base_path, "global_shift.pkl"))
    yield_lambda = joblib.load(os.path.join(base_path, "yield_lambda.pkl"))
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

def boxcox_transform(val, lam):
    return (val ** lam - 1) / lam if lam != 0 else np.log(val)

def inverse_boxcox(val, lam):
    safe_val = np.maximum(val * lam + 1, 1e-6)
    return np.power(safe_val, 1 / lam)

def predict_yield_streamlit(crop, district, season, area):
    input_dict = dict.fromkeys(X_columns, 0)
    area_input = area + global_shift
    area_boxcox = boxcox_transform(area_input, global_area_lambda)
    input_dict['Area_boxcox'] = area_boxcox

    if f'Crop_{crop}' in X_columns:
        input_dict[f'Crop_{crop}'] = 1
    if f'District_Name_{district}' in X_columns:
        input_dict[f'District_Name_{district}'] = 1
    if season != 'Kharif':
        season_col = f'Season_{season}'
        if season_col in X_columns:
            input_dict[season_col] = 1

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=X_columns, fill_value=0)
    input_scaled = scaler.transform(input_df)
    pred_transformed = model.predict(input_scaled)[0]
    return inverse_boxcox(pred_transformed, yield_lambda)

# Streamlit UI
st.set_page_config(page_title="Agricultural Yield Prediction", layout="centered")

st.title("🌾 Agricultural Yield Prediction App")
st.markdown("Predict crop yield (tons/hectare) using ML for selected crops and regions in Andhra Pradesh.")

crop = st.selectbox("🌱 Select Crop", ['Maize', 'Rice', 'Groundnut', 'Dry chillies', 'Onion', 'Arhar/Tur'])
district = st.selectbox("🏞️ Select District", ['GUNTUR', 'KURNOOL', 'WEST GODAVARI', 'SPSR NELLORE', 'VIZIANAGARAM'])
season = st.selectbox("☀️ Select Season", ['Kharif', 'Rabi', 'Whole Year'])
area = st.number_input("📐 Enter Area (in hectares)", min_value=0.1, value=1.0, step=0.1)

if st.button("🔍 Predict Yield"):
    try:
        result = predict_yield_streamlit(crop, district, season, area)
        st.success(f"✅ Predicted Yield: **{result:.2f} tons/hectare**")
    except Exception as e:
        st.error(f"Prediction error: {e}")
