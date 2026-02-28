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

kidney_model = pickle.load(open("models/kidney_model.pkl", "rb"))
kidney_scaler = pickle.load(open("models/kidney_scaler.pkl", "rb"))
kidney_columns = pickle.load(open("models/kidney_columns.pkl", "rb"))

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
    ["Diabetes", "Heart Disease","Liver Disease", "Kidney Disease"]
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
# ==============================
# Kidney Disease Section
# ==============================
if disease == "Kidney Disease":

    st.markdown('<div class="section-title">Kidney Disease Prediction</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120)
        bp = st.number_input("Blood Pressure", 50, 200)
        sg = st.number_input("Specific Gravity", 1.0, 1.05, step=0.001)
        al = st.number_input("Albumin (0-5)", 0, 5)
        su = st.number_input("Sugar (0-5)", 0, 5)
        rbc = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])
        pc = st.selectbox("Pus Cell", ["Normal", "Abnormal"])
        pcc = st.selectbox("Pus Cell Clumps", ["Present", "Not Present"])
        ba = st.selectbox("Bacteria", ["Present", "Not Present"])
        bgr = st.number_input("Blood Glucose Random", 0, 500)
        bu = st.number_input("Blood Urea", 0.0, 300.0)

    with col2:
        sc = st.number_input("Serum Creatinine", 0.0, 20.0)
        sod = st.number_input("Sodium", 0.0, 200.0)
        pot = st.number_input("Potassium", 0.0, 10.0)
        hemo = st.number_input("Hemoglobin", 0.0, 20.0)
        pcv = st.number_input("Packed Cell Volume", 0.0, 60.0)
        wc = st.number_input("White Blood Cell Count", 0, 20000)
        rc = st.number_input("Red Blood Cell Count", 0.0, 10.0)
        htn = st.selectbox("Hypertension", ["Yes", "No"])
        dm = st.selectbox("Diabetes Mellitus", ["Yes", "No"])
        cad = st.selectbox("Coronary Artery Disease", ["Yes", "No"])
        appet = st.selectbox("Appetite", ["Good", "Poor"])
        pe = st.selectbox("Pedal Edema", ["Yes", "No"])
        ane = st.selectbox("Anemia", ["Yes", "No"])

    # ======================
    # Convert categorical
    # ======================
    rbc = 1 if rbc == "Normal" else 0
    pc = 1 if pc == "Normal" else 0
    pcc = 1 if pcc == "Present" else 0
    ba = 1 if ba == "Present" else 0

    htn = 1 if htn == "Yes" else 0
    dm = 1 if dm == "Yes" else 0
    cad = 1 if cad == "Yes" else 0
    appet = 1 if appet == "Good" else 0
    pe = 1 if pe == "Yes" else 0
    ane = 1 if ane == "Yes" else 0

    # ======================
    # Prediction
    # ======================
    if st.button("Predict Kidney Disease Risk"):

        input_data = {
            "age": age,
            "bp": bp,
            "sg": sg,
            "al": al,
            "su": su,
            "rbc": rbc,
            "pc": pc,
            "pcc": pcc,
            "ba": ba,
            "bgr": bgr,
            "bu": bu,
            "sc": sc,
            "sod": sod,
            "pot": pot,
            "hemo": hemo,
            "pcv": pcv,
            "wc": wc,
            "rc": rc,
            "htn": htn,
            "dm": dm,
            "cad": cad,
            "appet": appet,
            "pe": pe,
            "ane": ane
        }

        input_df = pd.DataFrame([input_data])

        # Ensure same column order as training
        input_df = input_df.reindex(columns=kidney_columns, fill_value=0)

        input_scaled = kidney_scaler.transform(input_df)
        prediction = kidney_model.predict(input_scaled)

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error("⚠️ High Risk of Chronic Kidney Disease")
        else:
            st.success("✅ Low Risk of Kidney Disease")