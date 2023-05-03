import pickle
import sklearn
import streamlit as st
import pandas as pd
import numpy as np

def load_model():
    loaded_model = pickle.load(open("house_price.pkl", 'rb'))
    return loaded_model
  

st.title('House Pricing Prediction Web App')
st.write('This is a testing website for house pricing prediction.\
           Just for testing and practice web deployment for my own use. ')

area = st.sidebar.slider(label = 'house area', min_value = 0, max_value = 10000,
                         value = 9000, step = 1)
bedrooms = st.sidebar.slider(label = 'Number of bedrooms', min_value = 0, max_value = 10,
                             value = 5, step = 1)

features = {
  'area': area, 'bedrooms':bedrooms}

features_df = pd.DataFrame([features])

st.table(features_df)

if st.button('Predict'):
   load = load_model()
   prediction = load_model.predict(input)
   st.write('Based on features values, the house price is ' + str(int(prediction)))
