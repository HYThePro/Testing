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
        return (val-(5.134663e+03))/(2.199482e+03)
    elif type=="Bedrooms":
        return (val-(2.990826e+00))/(7.325388e-01)
    elif type=="Bathrooms"
        return (val-(1.295872e+00))/(5.041856e-01)	
    elif type=="stories"
        return (val-(1.811927e+00))/(8.494169e-01)	
    elif type=="Mainroad"
        return (val-(8.555046e-01))/(3.515914e-01)	
    elif type=="Guestroom"
        return (val-(1.674312e-01))/(3.733604e-01)
    elif type=="Basement"
        return (val-(3.463303e-01))/(4.758000e-01)	
    elif type=="Hot water heating"
        return (val-(4.587156e-02))/(2.092065e-01)
    elif type=="Air conditioning"
        return (val-(3.325688e-01))/(4.711335e-01)
    elif type=="Parking"
        return (val-(7.133028e-01))/(8.819844e-01)
    elif type=="Prefarea"
        return (val-(2.293578e-01))/(4.204198e-01)
    elif type=="Furnished"
        retun (val-(2.591743e-01))/(4.381815e-01)
    elif type=="Semi-furnished"
        return (val-(4.174312e-01))/(4.931353e-01)
    elif type=="Unfurnished"
        return (val-(3.233945e-01))/(4.677718e-01)
  
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

area = st.sidebar.slider(label = 'house area', min_value = 1650, max_value = 16200,
                         value = 3000, step = 1)
bedrooms = st.sidebar.slider(label = 'Number of bedrooms', min_value = 1, max_value = 6,
                             value = 3, step = 1)
bathroom = st.sidebar.slider(label = 'Number of bathroom', min_value = 1, max_value = 4,
                             value = 2, step = 1)
stories = st.sidebar.slider(label = 'Number of stories', min_value = 1, max_value = 4,
                             value = 3, step = 1)
mainroad = st.sidebar.slider(label = 'Has mainroad?', min_value = 0, max_value = 1,
                             value = 1, step = 1)
guestroom = st.sidebar.slider(label = 'Has guestroom?', min_value = 0, max_value = 1,
                             value = 0, step = 1)
basement = st.sidebar.slider(label = 'Has basement', min_value = 0, max_value = 1,
                             value = 1, step = 1)
hotwaterheating = st.sidebar.slider(label = 'Has hotwaterheating?', min_value = 0, max_value = 1,
                             value = 0, step = 1)
airconditioning = st.sidebar.slider(label = 'Has airconditioning?', min_value = 0, max_value = 1,
                             value = 0, step = 1)
parking = st.sidebar.slider(label = 'Number of parking', min_value = 0, max_value = 3,
                             value = 0, step = 1)
prefarea = st.sidebar.slider(label = 'Prefer area', min_value = 0, max_value = 1,
                             value = 0, step = 1)
furnished = st.sidebar.slider(label = 'furnished?', min_value = 0, max_value = 1,
                             value = 0, step = 1)
semi-furnished = st.sidebar.slider(label = 'semi-furnished?', min_value = 0, max_value = 1,
                             value = 0, step = 1)
unfurnished = st.sidebar.slider(label = 'unfurnished?', min_value = 0, max_value = 1,
                             value = 0, step = 1)


features = {
  'area': area, 'bedrooms':bedrooms}

adjusted_features=[normalize("Area",area),normalize("Bedrooms",bedrooms),normalize("Bathrooms",bathroom),normalize("Mainroad",mainroad),
                   normalize("Guestroom",guestroom),normalize("Basement",basement),normalise("Hot water heating",hotwaterheating),
                   normalize("Air conditioning", airconditioning), normalize("Parking",parking),normalize("Prefarea", prefarea),
                   normalize("Furnished",furnished),normlaize("Semi-furnished",semi-furnished),normalize("Unfurnished",unfurnished)]

input=np.array(adjusted_features).reshape(-1, 1)

features_df  = pd.DataFrame([features])
st.table(features_df)

if st.button('Predict'):
   load = load_model()
   prediction = load.predict(input)
   st.write('Based on features values, the house price is ' + str(int(prediction)))
