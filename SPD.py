#!/usr/bin/env python
# coding: utf-8

# In[13]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved model
with open('xgb_reg_model.pkl', 'rb') as file:
    xgb_reg_model = pickle.load(file)

# Streamlit app
st.title("Solar Power Generation Prediction")

# Input features
st.sidebar.header("Input Features")
# Inputs for key features
distance_to_solar_noon = st.sidebar.number_input("Distance to Solar Noon (radians)", min_value=0.0, max_value=2.0, step=0.01)
temperature = st.sidebar.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, step=0.1)
sky_cover = st.sidebar.selectbox("Sky Cover (0-4)", options=[0, 1, 2, 3, 4])
wind_speed = st.sidebar.number_input("Wind Speed (m/s)", min_value=0.0, max_value=50.0, step=0.1)

# Predict using placeholder values for unused features
unused_defaults = [0, 0, 0, 0, 0]  # Adjust these defaults if necessary
input_data = np.array([[distance_to_solar_noon, temperature, sky_cover, wind_speed] + unused_defaults])
if st.sidebar.button("Predict Power Generation"):
    prediction = xgb_reg_model.predict(input_data)
    st.markdown(
        f"""
        <div style="background-color:#f9f9f9;padding:20px;border-radius:10px;border:1px solid #ddd;text-align:center">
            <h2 style="color:#4CAF50;">Predicted Power Generation</h2>
            <p style="font-size:24px;color:#333;">{prediction[0]:.2f} Joules</p>
        </div>
        """, 
        unsafe_allow_html=True
    )






# In[ ]:




