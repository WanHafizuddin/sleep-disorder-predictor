import streamlit as st
import pandas as pd
import pickle

st.cache_resource.clear()

# --- Page Config ---
st.set_page_config(
    page_title="Sleep Disorder Predictor",
    page_icon="😴",
    layout="centered"
)

# --- Load Model ---
@st.cache_resource
def load_model():
    with open('sleep_disorder_model.pkl', 'rb') as f:
        return pickle.load(f)
model = load_model()

# --- Title ---
st.title("😴 Sleep Disorder Predictor")
st.markdown("Fill in your health and lifestyle details to predict your sleep disorder risk.")
st.divider()

# --- Input Form ---
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 80, 30)
    sleep_duration = st.slider("Sleep Duration (hours)", 4.0, 10.0, 7.0, step=0.1)
    quality_of_sleep = st.slider("Quality of Sleep (1-10)", 1, 10, 7)
    physical_activity = st.slider("Physical Activity Level (min/day)", 0, 120, 45)
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)

with col2:
    bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
    heart_rate = st.number_input("Heart Rate (bpm)", 40, 120, 72)
    daily_steps = st.number_input("Daily Steps", 0, 20000, 7000, step=500)
    bp_systolic = st.number_input("Blood Pressure Systolic", 80, 200, 120)
    bp_diastolic = st.number_input("Blood Pressure Diastolic", 50, 130, 80)

st.divider()

# --- Encode Inputs ---
gender_encoded = 1 if gender == "Male" else 0

bmi_map = {"Underweight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}
bmi_encoded = bmi_map[bmi_category]

# --- Predict ---
if st.button("🔍 Predict Sleep Disorder", use_container_width=True, type="primary"):

    input_data = pd.DataFrame([{
        'Gender': gender_encoded,
        'Age': age,
        'Sleep Duration': sleep_duration,
        'Quality of Sleep': quality_of_sleep,
        'Physical Activity Level': physical_activity,
        'Stress Level': stress_level,
        'BMI Category': bmi_encoded,
        'Heart Rate': heart_rate,
        'Daily Steps': daily_steps,
        'BP Systolic': bp_systolic,
        'BP Diastolic': bp_diastolic
    }])

    prediction = model.predict(input_data)[0]

    st.divider()

    if prediction == "None":
        st.success("✅ No Sleep Disorder Detected")
        st.markdown("Your sleep health looks good! Keep maintaining a healthy lifestyle.")

    elif prediction == "Insomnia":
        st.warning("⚠️ Insomnia Detected")
        st.markdown("""
        **Insomnia** means difficulty falling or staying asleep.  
        **Suggestions:**
        - Reduce stress and screen time before bed
        - Maintain a consistent sleep schedule
        - Avoid caffeine in the evening
        """)

    elif prediction == "Sleep Apnea":
        st.error("🚨 Sleep Apnea Detected")
        st.markdown("""
        **Sleep Apnea** means breathing repeatedly stops during sleep.  
        **Suggestions:**
        - Consult a doctor for proper diagnosis
        - Maintain a healthy BMI
        - Avoid sleeping on your back
        """)

    st.caption("⚠️ This is an ML prediction, not a medical diagnosis. Please consult a doctor.")

# --- Footer ---
st.divider()
st.caption("Built with Streamlit & scikit-learn | Sleep Health & Lifestyle Dataset")