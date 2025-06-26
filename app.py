import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("predict_salary.pkl")
scaler = joblib.load("scaler.pkl")


# Design the layout of our Basuc app
st.set_page_config(page_title = "Salary Predictor",layout = 'centered')
st.title("Salary Predictor App")
st.subheader("Predict your salary based on the number of years of experience")
st.write("select the years of experience to see the estimated salary")

# Create a dropdown for the years of experience
years = [x for x in range(0,20)]
years_exp  = st.selectbox("Years of Experience:",years)

# predict the salary

if st.button("Predict Salary"):
    input_data = np.array([[years_exp]])
    input_scaled = scaler.transform(input_data)
    predicted_salary = model.predict(input_scaled)
    st.success(f"Estimated salary is: ₹ {predicted_salary[0][0]:,.2f}")

    