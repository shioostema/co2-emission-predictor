import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Regional CO2 Emissions Predictor", layout="centered")

st.title("🌍 Regional CO2 Emissions Predictor")
st.write("This web application predicts average CO2 emissions based on SO2 and NOx hourly metrics.")

# Load the scaler and model
try:
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('co2_model.pkl')
    st.success("Model & Scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or scaler. Make sure 'scaler.pkl' and 'co2_model.pkl' are in the same folder as this script. Error: {e}")

st.subheader("Input Environmental Metrics")

# User Inputs
so2_input = st.number_input("Average SO2 Emissions (lbs/MWh)", min_value=0.0, max_value=50.0, value=2.0, step=0.1)
nox_input = st.number_input("Average NOx Emissions (lbs/MWh)", min_value=0.0, max_value=50.0, value=1.5, step=0.1)

if st.button("Predict CO2 Average"):
    # Prepare the input for scaling
    input_features = np.array([[so2_input, nox_input]])
    input_scaled = scaler.transform(input_features)

    # Predict
    prediction = model.predict(input_scaled)[0]

    st.markdown("--- ")
    st.metric(label="Predicted CO2 Average Emissions (lbs/MWh)", value=f"{prediction:.2f}")