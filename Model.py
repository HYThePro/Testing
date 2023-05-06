import pickle
import sklearn
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pycaret.regression import load_model, predict_model

def predict_rating(model, df):
    
    predictions_data = predict_model(estimator = model, data = df)
    
    return predictions_data['Label'][0]

def load_model():
    loaded_model = pickle.load(open("UMHackathon_model.pkl", 'rb'))
    return loaded_model

def normalize(type,val):
    if type=="Funding":
        return (val-(1.373136e+06))/(1.645224e+07)
    elif type=="Revenue":
        return (val-(3.460575e+06))/(3.595783e+07)
    elif type=="ebit":
        return (val-(-1.068171e+05))/(7.485605e+06)	
    elif type=="E6":
        return (val-(8.060758))/(38.771438)	
    elif type=="E12":
        return (val-(25.505254))/(79.271844)	
    elif type=="Founders":
        return (val-(2.428510))/(2.629927)
    elif type=="Rounds":
        return (val-(0.203746))/(0.650948)	
    elif type=="Shareholder":
        return (val-(6.439745))/(14.708537)
    elif type=="Median":
        return (val-(42.792926))/(31.694526)
    
funding = st.number_input(label='Total funding')
revenue = st.number_input(label='Revenue')
EBIT = st.number_input(label='EBIT')
e6 = st.number_input(label='Employee Growth (6 months)')
e12 = st.number_input(label='Employee Growth (12 months)')
founders = st.number_input(label='Number of founders')
rounds = st.number_input(label='Number of funding rounds')
shareholders = st.number_input(label='Number of shareholders')
median = st.number_input(label='Median share')

features = {
  'Funding': funding, 'Revenue':revenue, 'ebit':EBIT,
  'E6':e6, 'E12':e12, 'Founders':founders, 'Rounds':rounds,
  'Shareholder':shareholders, 'Median':median}

adjusted_features=[normalize("Funding",funding),normalize("Revenue",revenue),normalize("ebit",EBIT),
                   normalize("E6",e6),normalize("E12",e12),normalize("Founders",founders),
                   normalize("Rounds", rounds), normalize("Shareholder",shareholders),normalize("Median", median)]

input=np.array(adjusted_features).reshape(1, -1)


if st.button('Predict'):
   load = load_model()
   prediction = load.predict_rating(load, pd.DataFrame(input, colums=features.keys()))
   st.write('Based on features values, the house price is ' + str(int(prediction)))
