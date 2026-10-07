import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Model Bundle
bundle = joblib.load("Model/diabetes_model_bundle.joblib")

model = bundle["model"]
imputer = bundle["imputer"]
scaler = bundle["scaler"]
features = bundle["features"]
threshold = bundle["threshold"]


# Page Configuration
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)


# Title
st.title("🩺 Diabetes Prediction")
st.write("Enter the patient's details to predict diabetes risk.")


# Input Fields
Pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=1
)

Glucose = st.number_input(
    "Glucose",
    min_value=0,
    max_value=250,
    value=120
)

BloodPressure = st.number_input(
    "Blood Pressure",
    min_value=0,
    max_value=150,
    value=70
)

SkinThickness = st.number_input(
    "Skin Thickness",
    min_value=0,
    max_value=100,
    value=20
)

Insulin = st.number_input(
    "Insulin",
    min_value=0,
    max_value=900,
    value=80
)

BMI = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

DiabetesPedigreeFunction = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

Age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=30
)


# Prediction Button
if st.button("Predict Diabetes"):

    input_data = pd.DataFrame(
        [[
            Pregnancies,
            Glucose,
            BloodPressure,
            SkinThickness,
            Insulin,
            BMI,
            DiabetesPedigreeFunction,
            Age
        ]],
        columns=features
    )

    # Replace medically invalid zero values with NaN
    zero_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    input_data[zero_columns] = input_data[zero_columns].replace(
        0, np.nan
    )

    # Imputation
    input_imputed = imputer.transform(input_data)

    # Scaling
    input_scaled = scaler.transform(input_imputed)

    # Prediction Probability
    probability = model.predict_proba(input_scaled)[0][1]

    # Final prediction using optimized threshold
    prediction = int(probability >= threshold)


    # Display Result
    st.subheader("Prediction Result")

    st.write(f"Diabetes Probability: **{probability:.2%}**")
    st.write(f"Decision Threshold: **{threshold:.2f}**")

    if prediction == 1:
        st.error("⚠️ Higher Risk of Diabetes")
    else:
        st.success("✅ Lower Risk of Diabetes")

    st.info(
        "This prediction is for educational and demonstration purposes only "
        "and should not be considered a medical diagnosis."
    )