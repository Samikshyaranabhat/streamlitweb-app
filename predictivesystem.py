import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open('C:/Users/Acer/Downloads/prediction.py/trained_model.sav', 'rb'))

# Function for prediction
def diabetes_prediction(input_data):
    new = np.asarray(input_data)  
    reshape = new.reshape(1, -1)
    pred = loaded_model.predict(reshape)
    if pred[0] == 0:
        return "The person is not diabetic."
    else:
        return "The person is diabetic."

# Main function for the Streamlit app
def main():
    st.title('Diabetes Prediction System')

    # Input data from the user
    Pregnancies = st.number_input("Number of Pregnancies", min_value=0)
    Glucose = st.number_input("Glucose Level", min_value=0)
    BloodPressure = st.number_input("Blood Pressure value", min_value=0)
    SkinThickness = st.number_input("Skin Thickness value", min_value=0)
    Insulin = st.number_input("Insulin Level", min_value=0)
    BMI = st.number_input("BMI value", min_value=0.0, format="%.2f")
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function value", min_value=0.0, format="%.3f")
    Age = st.number_input("Age of the Person", min_value=0)

    # Prediction trigger
    diagnosis = ''
    if st.button("Diabetes Test Result"):
        input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        diagnosis = diabetes_prediction(input_data)

    st.success(diagnosis)


if __name__ == '__main__':
    main()
