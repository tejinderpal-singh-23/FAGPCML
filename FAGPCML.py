import joblib
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np

# Load models
RF = joblib.load('RF.joblib')
DTR = joblib.load('DTR.joblib')
SVR = joblib.load('SVR.joblib')
GBR = joblib.load('GBR.joblib')
KNN = joblib.load('KNN.joblib')
ETR = joblib.load('ETR.joblib')
BG = joblib.load('BG.joblib')
ADA = joblib.load('ADA.joblib')
XGB = joblib.load('XGB.joblib')
MLP = joblib.load('MLP.joblib')
LR = joblib.load('LR.joblib')


st.write('Geopolymer Concrete Compressive Strength predictor:')

# Dropdown menu for model selection
selected_model = st.selectbox(
    'Select ML model to predict:',
    ('RF_model', 'DTR_model', 'SVR_model', 'GBR_model', 'KNN_model', 'ETR_model', 'BG_model', 'ADA_model', 'XGB_model', 'MLP_model', 'LR_model')
)

# Input fields
Fine_aggregates_content = st.number_input('Fine aggregate content in kg/m3')
Coarse_aggregates_content = st.number_input('Coarse aggregate content in kg/m3')
Molarity = st.number_input('Molarity of NaoH solution')
Sodium_hydroxide_content = st.number_input('Sodium hydroxide solution content in kg/m3')
Sodium_silicate_content = st.number_input('Sodium silicate solution content in kg/m3')
Fly_ash_content = st.number_input('Fly Ash content in kg/m3')
Water_content = st.number_input('Water content in kg/m3')
Superplasticizer_content = st.number_input('Superplasticizer content in kg/m3')
Temperature = st.number_input('Curing temperature in degree centigrates')
Curing_age = st.number_input('Curing age in days')

        
input1 = [Fine_aggregates_content, Coarse_aggregates_content, Molarity, Sodium_hydroxide_content, Sodium_silicate_content, Fly_ash_content, Water_content, Superplasticizer_content, Temperature, Curing_age]
input1 = np.array(input1).reshape(1, -1)
    
if selected_model == 'RF_model':
    CS = RF.predict(input1)
    st.write("Predicted compressive strength is: " + str(CS) + " MPa")
elif selected_model == 'DTR_model':
    CS = DTR.predict(input1)
    st.write("Predicted compressive strength is: " + str(CS) + " MPa")
elif selected_model == 'SVR_model':
    CS = SVR.predict(input1)
    st.write("Predicted compressive strength is: " + str(CS) + " MPa")
elif selected_model == 'GBR_model':
    CS = GBR.predict(input1)
    st.write("Predicted compressive strength is: " + str(CS) + " MPa")
elif selected_model == 'KNN_model':
    CS = KNN.predict(input1)
    st.write("Predicted compressive strength is: " + str(CS) + " MPa")
elif selected_model == 'ETR_model':
    CS = ETR.predict(input1)
    st.write("Predicted compressive strength is: " + str(CS) + " MPa")
elif selected_model == 'BG_model':
    CS = BG.predict(input1)
    st.write("Predicted compressive strength is: " + str(CS) + " MPa")
elif selected_model == 'ADA_model':
    CS = ADA.predict(input1)
    st.write("Predicted compressive strength is: " + str(CS) + " MPa")
elif selected_model == 'XGB_model':
    CS = XGB.predict(input1)
    st.write("Predicted compressive strength is: " + str(CS) + " MPa")
elif selected_model == 'MLP_model':
    CS = MLP.predict(input1)
    st.write("Predicted compressive strength is: " + str(CS) + " MPa")
elif selected_model == 'LR_model':
    CS = LR.predict(input1)
    st.write("Predicted compressive strength is: " + str(CS) + " MPa")
else:
    st.write("Error in inputs: Compressive strength could not be predicted.")

st.write("Note: The predicted values are based on machine learning models developed by the author. "
         "The research is in initial stages and the values are just for giving a rough idea of the properties of Fly ash based geopolymer concrete. "
         "The results shall not be considered as final and experimental assessment of properties shall be done in practice.")

image1 = Image.open('developed_comb.png')
st.image(image1)
