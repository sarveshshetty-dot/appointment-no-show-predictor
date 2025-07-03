import streamlit as st 
import pandas as pd 
from data_preprocessing import load_and_preprocess_data 
from model import train_model, evaluate_model 
from intervention import suggest_intervention 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier

#Title and Description

st.title("🏥 Appointment No-Show Predictor") 
st.markdown(""" This application predicts whether a patient will miss their appointment based on:
Age, Gender, Appointment Type
Lead days between booking and appointment
Previous no-shows
Weather condition
It also suggests an appropriate intervention strategy to reduce no-shows. """)

#Load and display the dataset

st.header("📊 Dataset Preview") 
df = pd.read_csv("appointments.csv", parse_dates=['booking_date', 'appointment_date']) 
st.dataframe(df.head(10))

#Train model

st.header("🚀 Model Training and Prediction") 
X_train, X_test, y_train, y_test = load_and_preprocess_data("appointments.csv") 
model = train_model(X_train, y_train) 
probs = evaluate_model(model, X_test, y_test)

#Show predictions and interventions

st.subheader("📌 Predictions with Intervention Suggestions") 
for i, prob in enumerate(probs[:50]): 
    action = suggest_intervention(prob) 
    st.write(f"Patient {i+1}: No-show probability = {prob:.2f} → {action}")

#Summary Section

st.header("📈 Summary") 
total_high_risk = sum([1 for p in probs if p > 0.7]) 
total_medium_risk = sum([1 for p in probs if 0.4 < p <= 0.7]) 
total_low_risk = sum([1 for p in probs if p <= 0.4])

st.markdown(f"""

🔴 High Risk (Call): {total_high_risk}

🟡 Medium Risk (Send SMS): {total_medium_risk}

🟢 Low Risk (No Action): {total_low_risk} """)