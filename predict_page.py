import streamlit as st
import requests
import pandas as pd
import numpy as np
import pickle

# Load the pre-fitted scaler
try:
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except EOFError:
    st.error("Failed to load the scaler. The file might be corrupted or empty.")

# Function to display the prediction page
def show_predict_page():
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

    # Combine bmi and HbA1c_level for normalization
    combined_features = np.array([[bmi, HbA1c_level]])  # Combine both features into one array
    normalized_values = scaler.transform(combined_features)  # Normalize both features

    # Extract normalized values
    bmi_normalized = normalized_values[0, 0]  # First column for bmi
    HbA1c_normalized = normalized_values[0, 1]  # Second column for HbA1c

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
    if st.button('Predict'):
        try:
            response = requests.post('https://diabetes-prediction-app-o6yeqqeyiuwfac9zowfatq.streamlit.app/predict', json=df.to_dict(orient='records'))
            response.raise_for_status()  # Raise an error for bad status codes

            # Check if the response content is empty
            if not response.content:
                st.error("Received empty response from the server.")
            else:
                prediction = response.json().get('prediction', None)
                if prediction is not None:
                    if prediction[0] == 0:
                        st.success('The person does not have diabetes.')
                    else:
                        st.error('The person has diabetes.')
                else:
                    st.error('Invalid response from the server.')
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
        except ValueError:
            st.error("Failed to decode JSON response from the server.")

# Run the prediction page
if __name__ == '__main__':
    show_predict_page()