import streamlit as st
import pickle
import numpy as np 
import sys
import pandas as pd
from tensorflow.keras.models import load_model
classifier=load_model('model.h5')
st.title('Crop Yield Prediction')
st.write('this is a website curated for farmers')
bg_img='''
<style>
.stApp{
background-image:url('https://i0.wp.com/33.media.tumblr.com/40b9f1483a26a98f21945b42641ac2ea/tumblr_n3z57ezuBY1r2kyjno1_r3_500.gif');
background-size:cover;
}
'''
st.markdown(bg_img,unsafe_allow_html=True)

def yield_pre():
    crop_options = [
        "Arecanut", "Arhar/Tur", "Castor seed", "Coconut", "Cotton(lint)", "Dry chillies",
        "Gram", "Jute", "Linseed", "Maize", "Mesta", "Niger seed", "Onion", "Other Rabi pulses",
        "Potato", "Rapeseed & Mustard", "Rice", "Sesamum", "Small millets", "Sugarcane", 
        "Sweet potato", "Tapioca", "Tobacco", "Turmeric", "Wheat", "Bajra", "Black pepper",
        "Cardamom", "Coriander", "Garlic", "Ginger", "Groundnut", "Horse-gram", "Jowar", "Ragi",
        "Cashewnut", "Banana", "Soyabean", "Barley", "Khesari", "Masoor", "Moong(Green Gram)",
        "Other Kharif pulses", "Safflower", "Sannhamp", "Sunflower", "Urad",
        "Peas & beans (Pulses)", "Other oilseeds", "Other Cereals", "Cowpea(Lobia)",
        "Oilseeds total", "Guar seed", "Other Summer Pulses", "Moth"
    ]
    
    season_options = [
        "Whole Year", "Kharif", "Rabi", "Autumn", "Summer", "Winter"
    ]
    
    state_options = [
        "Assam", "Karnataka", "Kerala", "Meghalaya", "West Bengal", "Puducherry", "Goa",
        "Andhra Pradesh", "Tamil Nadu", "Odisha", "Bihar", "Gujarat", "Madhya Pradesh", 
        "Maharashtra", "Mizoram", "Punjab", "Uttar Pradesh", "Haryana", "Himachal Pradesh", 
        "Tripura", "Nagaland", "Chhattisgarh", "Uttarakhand", "Jharkhand", "Delhi", 
        "Manipur", "Jammu and Kashmir", "Telangana", "Arunachal Pradesh", "Sikkim"
    ]

    Crop = st.selectbox("Select Crop", crop_options)
    Season = st.radio("Select Season", season_options)
    State = st.selectbox("Select State", state_options)
    Crop_Year = st.slider("Year", min_value=1997, max_value=2020, value=2000)
    Area = st.slider("Area (hectares)", min_value=0.5, max_value=1000000.0, value=100.0)
    Production = st.slider("Production", min_value=0.0, max_value=1e8, value=1000.0)
    Annual_Rainfall = st.slider("Annual Rainfall (mm)", min_value=300.0, max_value=3000.0, value =2051.4)
    Fertilizer = st.slider("Fertilizer (kg/ha)", min_value=0.0, max_value=1e7, value=7024878.38)
    Pesticide = st.slider("Pesticide (kg/ha)", min_value=0.0, max_value=50000.0, value=22882.34)

    crop_map = {crop: idx for idx, crop in enumerate(crop_options)}
    state_map = {state: idx for idx, state in enumerate(state_options)}
    season_map = {season: idx for idx, season in enumerate(season_options)}

    Crop = crop_map[Crop]
    Season = season_map[Season]
    State = state_map[State]

    data = {
        'Crop': Crop,
        'State': State,
        'Season': Season,
        'Crop_Year': Crop_Year,
        'Area': Area,
        'Production': Production,
        'Annual_Rainfall': Annual_Rainfall,
        'Fertilizer': Fertilizer,
        'Pesticide': Pesticide
    }
    
    features = np.array([Crop, Crop_Year, Season, State, Area, Production, Annual_Rainfall, Fertilizer, Pesticide], dtype=np.float32)
    return features

inputs = yield_pre()

if st.button("Predict Yield"):
    inputs = inputs.reshape(1, -1)
    try:
        prediction = classifier.predict(inputs)
        if prediction>0:
            st.write('The crop can be grown')
        else:
            st.write(f"The crop is not suitable for the conditions given")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")