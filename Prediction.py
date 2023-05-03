from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

def predict(model, df):
  prediction = predict_model(estimator = model, data = df)
  return prediction['price'][0]

model = load_model('house_price')

st.title('House Pricing Prediction Web App')
st.write('This is a testing website for house pricing prediction.\
           Just for testing and practice web deployment for my own use. ')

area = st.sidebar.slider(label = 'house area', min_value = 0, max_value = 1000000,
                         value = 9999, step = 1)

features = {
  'area': area}

features_df = pd.DataFrame([Features])

st.table(features_df)

if(st.button('Predict'):
   prediction = predict(model, features_df)
   st.write('Based on features values, the house price is ' + str(int(prediction)))
