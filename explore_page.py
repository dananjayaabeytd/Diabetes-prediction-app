import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

# Load your dataset (Ensure this is your cleaned data)
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    
    # Convert categorical columns to numeric
    gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
    hypertension_map = {'No': 0, 'Yes': 1}
    heart_disease_map = {'No': 0, 'Yes': 1}
    smoking_history_map = {'never': 3, 'not current': 4, 'current': 2, 'former': 0, 'ever': 1}

    # Map values to numeric
    df['gender'] = df['gender'].map(gender_map)
    df['hypertension'] = df['hypertension'].map(hypertension_map)
    df['heart_disease'] = df['heart_disease'].map(heart_disease_map)
    df['smoking_history'] = df['smoking_history'].map(smoking_history_map)

    return df

df = load_data()

def show_explore_page():
    st.title("Explore Diabetes Data")
    
    # Navigation radio button to toggle between visualizations
    visualization_type = st.radio(
        "Select Visualization Type", 
        ('Overall Dataset Visualization', 'Patient’s Data Visualization')
    )

    if visualization_type == 'Overall Dataset Visualization':
        st.header("Overall Dataset Visualization")
        
        # Count Plot of Diabetes Cases
        st.write("### Distribution of Diabetes Cases")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, x='diabetes', ax=ax1)
        ax1.set_xticklabels(['No Diabetes', 'Diabetes'])
        st.pyplot(fig1)

        # Histogram of Age Distribution for Diabetic Patients
        st.write("### Age Distribution of Diabetic Patients")
        fig2, ax2 = plt.subplots()
        sns.histplot(df[df['diabetes'] == 1]['age'], bins=30, kde=True, ax=ax2)
        ax2.set_title('Age Distribution for Patients with Diabetes')
        ax2.set_xlabel('Age')
        ax2.set_ylabel('Frequency')
        st.pyplot(fig2)

        # Box Plot of BMI Distribution by Diabetes Status
        st.write("### BMI Distribution by Diabetes Status")
        fig3, ax3 = plt.subplots()
        sns.boxplot(data=df, x='diabetes', y='bmi', ax=ax3)
        ax3.set_xticklabels(['No Diabetes', 'Diabetes'])
        ax3.set_title('BMI Distribution by Diabetes Status')
        st.pyplot(fig3)

        # Box Plot of Blood Glucose Level Distribution by Diabetes Status
        st.write("### Blood Glucose Level Distribution by Diabetes Status")
        fig4, ax4 = plt.subplots()
        sns.boxplot(data=df, x='diabetes', y='blood_glucose_level', ax=ax4)
        ax4.set_xticklabels(['No Diabetes', 'Diabetes'])
        ax4.set_title('Blood Glucose Level Distribution by Diabetes Status')
        st.pyplot(fig4)

    elif visualization_type == 'Patient’s Data Visualization':
        st.header("Patient’s Data Visualization")

        # Visualization for Blood Glucose Level (User Input)
        user_blood_glucose = st.number_input('Enter Blood Glucose Level', min_value=0, max_value=300, value=100)
        fig5, ax5 = plt.subplots()
        ax5.bar(['Your Blood Glucose Level', 'Normal Range'], 
                 [user_blood_glucose, 140], color=['blue', 'orange'])  # 140 mg/dL is an average threshold for diabetes
        ax5.set_ylabel('Blood Glucose Level (mg/dL)')
        st.pyplot(fig5)

        # Visualization for BMI (User Input)
        user_bmi = st.number_input('Enter BMI', min_value=0.0, max_value=50.0, value=25.0)
        fig6, ax6 = plt.subplots()
        ax6.bar(['Your BMI', 'Normal Range'], 
                 [user_bmi, 24.9], color=['green', 'red'])  # 24.9 is the upper limit of the normal BMI range
        ax6.set_ylabel('BMI')
        st.pyplot(fig6)

        # Visualization for HbA1c Level (User Input)
        user_hba1c = st.number_input('Enter HbA1c Level', min_value=0.0, max_value=15.0, value=5.5)
        fig7, ax7 = plt.subplots()
        ax7.bar(['Your HbA1c Level', 'Normal Range'], 
                 [user_hba1c, 5.7], color=['purple', 'pink'])  # 5.7 is the upper limit of normal HbA1c
        ax7.set_ylabel('HbA1c Level (%)')
        st.pyplot(fig7)

# To run this page, you will call `show_explore_page()` from your main app file.
