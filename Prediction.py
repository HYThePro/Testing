import pickle
import sklearn
import streamlit as st
import pandas as pd
import numpy as np

def load_model():
    loaded_model = pickle.load(open("house_price.pkl", 'rb'))
    return loaded_model

def normalize(type,val):
    if type=="Area":
        return (val-5150.541284)/(2170.141023)
    elif type=="Bedrooms":
        return (val-2.965138)/(0.738064)
  
st.title('House Pricing Prediction Web App')
st.write("""
        This is a testing website for house pricing prediction.
        
        Just for testing and practice web deployment for my own use.""")

st.subheader("Calculation")
st.write("""
          Please enter your value after calculated based on the formula below
          
          Formula = (X-u)/s
          
          X = your value
          
          u = mean
          
          s = standard deviation""")
st.text("")
st.write("---")
st.text("")
st.subheader("Mean and standard deviation")
st.write("""
         Mean for area = 5150.541284, stadard deviation = 2170.141023
         
         Mean for bedrooms = 2.965138, standard deviation = 0.738064""")
st.text("")
st.write("---")
st.text("")

area = st.sidebar.slider(label = 'house area', min_value = 0, max_value = 10000,
                         value = 1000, step = 1)
bedrooms = st.sidebar.slider(label = 'Number of bedrooms', min_value = 0, max_value = 10,
                             value = 3, step = 1)

features = {
  'area': area, 'bedrooms':bedrooms}

adjusted_features=[normalize("Area",area),normalize("Bedrooms",bedrooms),]

input=np.array(adjusted_features).reshape(1,-1)

features_df  = pd.DataFrame([features])
st.table(features_df)

if st.button('Predict'):
   load = load_model()
   prediction = load.predict(input)
   st.write('Based on features values, the house price is ' + str(int(prediction)))
