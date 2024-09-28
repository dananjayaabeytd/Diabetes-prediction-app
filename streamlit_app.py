import streamlit as st
import requests
import pandas as pd
import numpy as np
import pickle

# Load the pre-fitted scaler
try:
    with open('./model/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except EOFError:
    st.error("Failed to load the scaler. The file might be corrupted or empty.")

# Custom CSS for styling
st.markdown("""
    <style>
    .stButton>button {
        border: none;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        cursor: pointer;
        border-radius: 12px;
            
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit interface
st.title('Diabetes Prediction')
st.markdown("### Please enter the following details:")




# User inputs
col1, col2 = st.columns(2)


with col1:
    hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
    heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
    blood_glucose_level = st.slider('Blood Glucose Level', min_value=0, max_value=300, value=100)
    bmi = st.slider('BMI', min_value=0.0, max_value=50.0, value=25.0)    
    

with col2:
    smoking_history = st.selectbox('Smoking History', ['never', 'not current', 'current', 'former', 'ever'])
    gender = st.selectbox('Gender', ['Female', 'Male', 'Other'])
    age = st.slider('Age', min_value=0, max_value=120, value=30)
    HbA1c_level = st.slider('HbA1c Level', min_value=0.0, max_value=15.0, value=5.5)

# Map user inputs to numerical values
gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
hypertension_map = {'No': 0, 'Yes': 1}
heart_disease_map = {'No': 0, 'Yes': 1}
smoking_history_map = {'never': 3, 'not current': 4, 'current': 2, 'former': 0, 'ever': 1}

# Normalize the bmi and HbA1c_level inputs
normalized_values = scaler.transform(np.array([[bmi, HbA1c_level]]))
bmi_normalized = normalized_values[0, 0]
HbA1c_normalized = normalized_values[0, 1]

# Create a DataFrame from user inputs
data = {
    'gender': [gender_map[gender]],
    'age': [age],
    'hypertension': [hypertension_map[hypertension]],
    'heart_disease': [heart_disease_map[heart_disease]],
    'smoking_history': [smoking_history_map[smoking_history]],
    'blood_glucose_level': [blood_glucose_level],
    'bmi_normalized': [bmi_normalized],
    'HbA1c_normalized': [HbA1c_normalized]
}
df = pd.DataFrame(data)

# Button to make prediction
# if st.button('Predict'):
#     # response = requests.post('http://127.0.0.1:5000/predict', json=df.to_dict(orient='records'))
#     response = requests.post('https://d-predict-app.streamlit.app/predict', json=df.to_dict(orient='records'))
#     prediction = response.json()['prediction']
    
#     if prediction[0] == 0:
#         st.success('The person does not have diabetes.')
#     else:
#         st.error('The person has diabetes.')

if st.button('Predict'):
    try:
        response = requests.post('https://d-predict-app.streamlit.app/predict', json=df.to_dict(orient='records'))
        
        # Log the response status and content
        st.write(f"Response status code: {response.status_code}")
        st.write(f"Response headers: {response.headers}")
        st.write(f"Response content: {response.content.decode('utf-8')}")
        
        # Check if the response is JSON
        if response.headers.get('Content-Type') == 'application/json':
            prediction = response.json().get('prediction')
            if prediction is not None:
                if prediction[0] == 0:
                    st.success('The person does not have diabetes.')
                else:
                    st.error('The person has diabetes.')
            else:
                st.error('Prediction key not found in the response.')
        else:
            st.error('The response is not in JSON format.')
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
    except ValueError as e:
        st.error(f"JSON decoding failed: {e}")