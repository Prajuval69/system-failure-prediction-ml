import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained model and scaler
model = joblib.load("failure_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="System Failure Prediction", layout="centered")

st.title("System Failure Prediction Dashboard")
st.markdown("Predict potential system failure using real-time performance metrics.")

st.sidebar.header("System Metrics Input")

cpu = st.sidebar.slider("CPU Usage (%)", 0, 100, 50)
ram = st.sidebar.slider("RAM Usage (%)", 0, 100, 50)
disk = st.sidebar.slider("Disk Usage (%)", 0, 100, 50)
network = st.sidebar.slider("Network Traffic", 0, 1000, 200)
errors = st.sidebar.slider("Error Count", 0, 20, 2)
temp = st.sidebar.slider("Temperature (°C)", 20, 100, 50)
uptime = st.sidebar.slider("Uptime (Hours)", 1, 500, 100)

if st.sidebar.button("Predict Failure"):

    input_data = np.array([[cpu, ram, disk, network, errors, temp, uptime]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ Failure Likely\n\nProbability: {probability * 100:.2f}%")
    else:
        st.success(f"✅ System Stable\n\nFailure Probability: {probability * 100:.2f}%")

    st.subheader("Feature Importance (Random Forest Model)")

    features = ["CPU Usage", "RAM Usage", "Disk Usage", "Network Traffic",
                "Error Count", "Temperature", "Uptime"]

    importances = model.feature_importances_

    fig, ax = plt.subplots()
    ax.barh(features, importances)
    ax.set_xlabel("Importance Score")
    ax.set_title("Key Factors Influencing System Failure")

    st.pyplot(fig)
