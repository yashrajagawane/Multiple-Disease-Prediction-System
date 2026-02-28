import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Multiple Disease Prediction System",
    layout="wide",
    page_icon="🩺"
)

# ==============================
# Load Models & Scalers
# ==============================
diabetes_model = pickle.load(open("models/diabetes_model.pkl", "rb"))
diabetes_scaler = pickle.load(open("models/diabetes_scaler.pkl", "rb"))

heart_model = pickle.load(open("models/heart_model.pkl", "rb"))
heart_scaler = pickle.load(open("models/heart_scaler.pkl", "rb"))
heart_columns = pickle.load(open("models/heart_columns.pkl", "rb"))

liver_model = pickle.load(open("models/liver_model.pkl", "rb"))
liver_scaler = pickle.load(open("models/liver_scaler.pkl", "rb"))
liver_columns = pickle.load(open("models/liver_columns.pkl", "rb"))

# ==============================
# Custom Styling
# ==============================
st.markdown("""
<style>
.big-title {
    font-size:45px !important;
    font-weight:900 !important;
    color:#2E86C1 !important;
}
.big-subtitle {
    font-size:22px !important;
    color:#AEB6BF !important;
    margin-top:-10px;
}
.section-title {
    font-size:32px !important;
    font-weight:700 !important;
    margin-top:20px;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# Sidebar
# ==============================
st.sidebar.title("🩺 Disease Selection")
disease = st.sidebar.selectbox(
    "Choose Disease",
    ["Diabetes", "Heart Disease","Liver Disease"]
)

# ==============================
# Header
# ==============================
st.markdown('<div class="big-title">🩺 Multiple Disease Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="big-subtitle">AI-Powered Health Risk Assessment Dashboard</div>', unsafe_allow_html=True)
st.divider()

# ==============================
# Diabetes Section
# ==============================
if disease == "Diabetes":

    st.markdown('<div class="section-title">Diabetes Prediction</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20)
        glucose = st.number_input("Glucose Level", 50, 300)
        blood_pressure = st.number_input("Blood Pressure", 40, 200)
        skin_thickness = st.number_input("Skin Thickness", 0, 100)

    with col2:
        insulin = st.number_input("Insulin Level", 0, 900)
        bmi = st.number_input("BMI", 10.0, 60.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 5.0)
        age = st.number_input("Age", 1, 120)

    st.divider()

    if st.button("Predict Diabetes Risk"):

        input_data = np.array([[pregnancies, glucose, blood_pressure,
                                skin_thickness, insulin, bmi, dpf, age]])

        input_scaled = diabetes_scaler.transform(input_data)
        prediction = diabetes_model.predict(input_scaled)

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error("⚠️ High Risk of Diabetes Detected")
        else:
            st.success("✅ Low Risk of Diabetes")

# ==============================
# Heart Disease Section
# ==============================
if disease == "Heart Disease":

    st.markdown('<div class="section-title">Heart Disease Prediction</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.number_input("Chest Pain Type (0-3)", 0, 3)
        trestbps = st.number_input("Resting Blood Pressure", 80, 200)
        chol = st.number_input("Cholesterol", 100, 600)

    with col2:
        fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl", ["Yes", "No"])
        thalach = st.number_input("Max Heart Rate Achieved", 60, 220)
        exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0)
        ca = st.number_input("Major Vessels (0-3)", 0, 3)

    if st.button("Predict Heart Disease Risk"):

        # Convert categorical
        sex = 1 if sex == "Male" else 0
        fbs = 1 if fbs == "Yes" else 0
        exang = 1 if exang == "Yes" else 0

        # Create input dictionary
        input_dict = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "ca": ca
        }

        input_df = pd.DataFrame([input_dict])

        # One-hot encode like training
        input_df = pd.get_dummies(input_df)

        # Match training columns
        input_df = input_df.reindex(columns=heart_columns, fill_value=0)

        # Scale
        input_scaled = heart_scaler.transform(input_df)

        prediction = heart_model.predict(input_scaled)

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error("⚠️ High Risk of Heart Disease Detected")
        else:
            st.success("✅ Low Risk of Heart Disease")

# ==============================
# Liver Disease Section
# ==============================
if disease == "Liver Disease":

    st.markdown('<div class="section-title">Liver Disease Prediction</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120)
        gender = st.selectbox("Gender", ["Male", "Female"])
        tot_bilirubin = st.number_input("Total Bilirubin", 0.0, 50.0)
        direct_bilirubin = st.number_input("Direct Bilirubin", 0.0, 20.0)
        tot_proteins = st.number_input("Total Proteins", 0.0, 2000.0)

    with col2:
        albumin = st.number_input("Albumin", 0.0, 100.0)
        ag_ratio = st.number_input("Albumin/Globulin Ratio", 0.0, 10.0)
        sgpt = st.number_input("SGPT", 0.0, 500.0)
        sgot = st.number_input("SGOT", 0.0, 500.0)
        alkphos = st.number_input("Alkaline Phosphotase", 0.0, 2000.0)

    # Convert gender
    gender = 1 if gender == "Male" else 0

    if st.button("Predict Liver Disease Risk"):

        input_data = {
            "age": age,
            "gender": gender,
            "tot_bilirubin": tot_bilirubin,
            "direct_bilirubin": direct_bilirubin,
            "tot_proteins": tot_proteins,
            "albumin": albumin,
            "ag_ratio": ag_ratio,
            "sgpt": sgpt,
            "sgot": sgot,
            "alkphos": alkphos
        }

        input_df = pd.DataFrame([input_data])
        input_df = input_df[liver_columns]

        input_scaled = liver_scaler.transform(input_df)
        prediction = liver_model.predict(input_scaled)

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error("⚠️ High Risk of Liver Disease")
        else:
            st.success("✅ Low Risk of Liver Disease")