import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

# Load the trained model pipeline
@st.cache_resource
def load_model():
    return joblib.load('diabetes_model.pkl')

model = load_model()

# Title and Info
st.title("🩺 Diabetes Risk Prediction App")
st.write("Enter the patient's medical details below to predict the likelihood of diabetes.")

st.markdown("---")

# User Inputs (split into two neat columns)
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=117)
    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, value=72)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=29)

with col2:
    insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=125)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=32.3, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.372, format="%.3f")
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=29, step=1)

st.markdown("---")

# Predict Button
if st.button("Predict Risk", type="primary", use_container_width=True):
    # Format input into dataframe matching feature names
    input_data = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age
    ]], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    # Make prediction & probability estimate
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100
    
    # Display Output
    st.subheader("Results")
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Diabetes** (Estimated Probability: {probability:.1f}%)")
        st.info("Consider consulting a healthcare professional for further clinical assessment.")
    else:
        st.success(f"✅ **Low Risk of Diabetes** (Estimated Probability: {probability:.1f}%)")
        st.info("The parameters indicate a low risk profile. Keep up a healthy lifestyle!")