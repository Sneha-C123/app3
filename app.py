import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

st.title("Heart Attack Risk Classification")

# load model safely
try:
    model = pickle.load(open('rf_model.pkl', 'rb'))
except:
    st.error("Model file not found!")
    st.stop()

# inputs
age = st.number_input('Age', min_value=20, max_value=100, value=25)
RestingBP = st.number_input('RestingBP', min_value=0, max_value=300, value=120)
Cholesterol = st.number_input('Cholesterol', min_value=0, max_value=600, value=200)
MaxHR = st.number_input('MaxHR', min_value=60, max_value=220, value=150)
Oldpeak = st.number_input('Oldpeak', min_value=-3.0, max_value=10.0, value=2.0)
FastingBS = st.selectbox('FastingBS', (0, 1))
gender = st.selectbox('Gender', ('M', 'F'))
ChestPainType = st.selectbox('ChestPainType', ('ATA', 'NAP', 'ASY', 'TA'))
RestingECG = st.selectbox('RestingECG', ('Normal', 'ST', 'LVH'))
ExerciseAngina = st.selectbox('ExerciseAngina', ('N', 'Y'))
ST_Slope = st.selectbox('ST_Slope', ('Up', 'Flat', 'Down'))

# encoding
Exercise_Angina = 1 if ExerciseAngina == 'Y' else 0
Sex_F = 1 if gender == 'F' else 0
Sex_M = 1 if gender == 'M' else 0

ChestPainType_dict = {'ASY': 3, 'NAP': 2, 'ATA': 1, 'TA': 0}
RestingECG_dict = {'Normal': 0, 'LVH': 1, 'ST': 2}
ST_Slope_dict = {'Up': 1, 'Flat': 2, 'Down': 3}

ChestPainType = ChestPainType_dict[ChestPainType]
RestingECG = RestingECG_dict[RestingECG]
ST_Slope = ST_Slope_dict[ST_Slope]

# dataframe
input_features = pd.DataFrame({
    'Age': [age],
    'RestingBP': [RestingBP],
    'Cholesterol': [Cholesterol],
    'FastingBS': [FastingBS],
    'MaxHR': [MaxHR],
    'Oldpeak': [Oldpeak],
    'Exercise_Angina': [Exercise_Angina],
    'Sex_F': [Sex_F],
    'Sex_M': [Sex_M],
    'Chest_PainType': [ChestPainType],
    'Resting_ECG': [RestingECG],
    'st_Slope': [ST_Slope]
})

# scaling
scaler = StandardScaler()
input_features[['Age', 'RestingBP', 'Cholesterol', 'MaxHR']] = scaler.fit_transform(
    input_features[['Age', 'RestingBP', 'Cholesterol', 'MaxHR']]
)

# prediction
if st.button('Predict'):
    prediction = model.predict(input_features)

    if prediction[0] == 1:
        st.error('⚠️ High risk of Heart Attack')
    else:
        st.success('😊 Low risk of Heart Attack')