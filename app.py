
import streamlit as st
import pandas as pd
import joblib 
import base64 

# Function to add background
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded_img = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_img}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add background image
add_bg_from_local("htt.webp")   # Change file name as needed


model = joblib.load("KNN_heart.pkl")
scaler =joblib.load("scaler.pkl")
expected_columns=joblib.load("columns.pkl")

st.title("Heart Stroke Prediction By Ajay❤️")
st.markdown("provide the following details")

age = st.slider("Age",18,100,40)
sex = st.selectbox("SEX",['M','F'])
chest_pain=st.selectbox("Chest Pain Type",["ATA","NAP","TA","ASY"])
resting_bp = st.number_input("Resting Blood Pressure(mm Hg)",80,200,120)
cholestrol = st.number_input("Cholestrol(mg/dL)",100,600,200)
fasting_bs=st.selectbox("Fasting Blood sugar< 120 mg/dL",[0,1])
resting_ecg=st.selectbox("Resting ECG",["Normal","ST","LVH"])
max_hr =st.slider("Max Heart Rate",60,220,150)
exercise_angina= st.selectbox("Exercise-Induced Angine",["Y","N"])
oldpeak = st.slider("Oldpeak(ST Depression)",0.0,6.0,1.0)
st_slope = st.selectbox("ST Slope",["Up","Flat","Down"])

if st.button("Predict"):
    raw_input ={
  'Age' : age,
  'Sex_'+ sex:1,
  'Oldpeak':oldpeak,
  'RestingBP': resting_bp,
  'Cholesterol': cholestrol,
  'FastingBS': fasting_bs,
   'MaxHR': max_hr,
   'RestingECG_'+ resting_ecg: 1 ,
  'ExerciseAngina_'+ exercise_angina:1,
  'ChestPainType_'+ chest_pain:1,
  'ST_Slope'+ st_slope:1,
}
input_df = pd.DataFrame([raw_input])

for col in expected_columns:
    if col not in input_df.columns:
        input_df[col]=0

input_df=input_df[expected_columns]

scaled_input= scaler.transform(input_df)
prediction=model.predict(scaled_input)[0]

if prediction==1:
    st.error("⚠️High Risk of Hear Disease")
else:
    st.success("👍Low Risk Of Heart Disease")