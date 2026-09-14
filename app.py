import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Diabetic Retinopathy Prediction",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# Load Model and Scaler
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("svm_retinopathy_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


model, scaler = load_artifacts()

# -----------------------------
# Title
# -----------------------------
st.title("🩺 Diabetic Retinopathy Prediction")

st.write(
    "Enter the patient's clinical information below "
    "to predict the risk of diabetic retinopathy."
)

st.divider()

# -----------------------------
# Input Section
# -----------------------------
st.subheader("Patient Information")

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=120.0,
    value=50.0,
    step=0.1
)

systolic_bp = st.number_input(
    "Systolic Blood Pressure",
    min_value=0.0,
    max_value=300.0,
    value=120.0,
    step=0.1
)

diastolic_bp = st.number_input(
    "Diastolic Blood Pressure",
    min_value=0.0,
    max_value=200.0,
    value=80.0,
    step=0.1
)

cholesterol = st.number_input(
    "Cholesterol",
    min_value=0.0,
    max_value=500.0,
    value=100.0,
    step=0.1
)

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict", use_container_width=True):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "age": [age],
        "systolic_bp": [systolic_bp],
        "diastolic_bp": [diastolic_bp],
        "cholesterol": [cholesterol]
    })

    # Apply same scaler used during training
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Probability
    probability = model.predict_proba(input_scaled)[0]

    # -----------------------------
    # Display Result
    # -----------------------------
    st.subheader("Prediction Result")

    if prediction == 1:

        confidence = probability[1] * 100

        st.error("⚠️ Retinopathy Detected")

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.warning(
            "The model predicts retinopathy based on the "
            "provided clinical features."
        )

    else:

        confidence = probability[0] * 100

        st.success("✅ No Retinopathy Detected")

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.info(
            "The model predicts no retinopathy based on the "
            "provided clinical features."
        )

    # -----------------------------
    # Show Input Data
    # -----------------------------
    st.subheader("Patient Input")

    display_data = pd.DataFrame({
        "Feature": [
            "Age",
            "Systolic BP",
            "Diastolic BP",
            "Cholesterol"
        ],
        "Value": [
            age,
            systolic_bp,
            diastolic_bp,
            cholesterol
        ]
    })

    st.table(display_data)

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "Machine Learning Model: Support Vector Machine (SVM)"
)

st.caption(
    "This application is for educational/research purposes "
    "and should not replace professional medical diagnosis."
)