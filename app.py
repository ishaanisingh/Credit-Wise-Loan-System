import streamlit as st
import pickle
import numpy as np

# 1. Load your saved model and scaler
with open('loan_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# 2. Build the UI
st.title("Credit-Wise: Loan Approval Predictor")
st.write("Enter the applicant's details below to predict loan approval.")

# Create input fields for the user
income = st.number_input("Applicant Income", min_value=0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850)
dti_ratio = st.number_input("Debt-to-Income (DTI) Ratio", min_value=0.0)

# 3. Create the Prediction Button
if st.button("Predict Loan Status"):
    
    # Remember your feature engineering! You must apply the same math here.
    credit_score_sq = credit_score ** 2
    dti_ratio_sq = dti_ratio ** 2
    
    # Arrange the inputs in the EXACT SAME ORDER as your training data columns
    # Example: [Applicant_Income, Credit_Score_Sq, DTI_Ratio_Sq, ...]
    user_input = np.array([[income, credit_score_sq, dti_ratio_sq]])
    
    # Scale the input
    user_input_scaled = scaler.transform(user_input)
    
    # Make the prediction
    prediction = model.predict(user_input_scaled)
    
    # Display the result
    if prediction[0] == 1:
        st.success("🎉 Loan Approved!")
    else:
        st.error("❌ Loan Denied.")