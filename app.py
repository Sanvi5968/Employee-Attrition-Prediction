import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)

# Load model and feature names
model = pickle.load(open('employee_attrition_model.pkl', 'rb'))
feature_names = pickle.load(open('feature_names.pkl', 'rb'))

st.markdown("""
<style>
.title { font-size: 42px; font-weight: bold; color: #1f4e79; }
.subtitle { font-size: 18px; color: #555; margin-bottom: 30px; }
.prediction-box { padding: 20px; border-radius: 10px; background-color: #ffffff;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">📊 Employee Attrition Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict whether an employee is likely to leave the company using Machine Learning.</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.slider('Age', 18, 60, 30)
    monthly_income = st.number_input('Monthly Income', 1000, 50000, 5000)
    years_at_company = st.slider('Years At Company', 0, 40, 5)

with col2:
    job_satisfaction = st.slider('Job Satisfaction', 1, 4, 3)
    overtime = st.selectbox('OverTime', ['Yes', 'No'])

overtime_val = 1 if overtime == 'Yes' else 0

# Build a dict with ALL expected features, defaulting to 0
input_dict = {col: 0 for col in feature_names}

# Fill in the ones the user actually provided
input_dict['Age'] = age
input_dict['MonthlyIncome'] = monthly_income
input_dict['YearsAtCompany'] = years_at_company
input_dict['JobSatisfaction'] = job_satisfaction
if 'OverTime_Yes' in input_dict:
    input_dict['OverTime_Yes'] = overtime_val

# Convert to DataFrame with correct column order
input_data = pd.DataFrame([input_dict])[feature_names]

st.markdown("---")

if st.button('Predict Attrition'):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)

    if prediction[0] == 1:
        st.error('⚠️ Employee is likely to leave the company.')
        st.metric("Attrition Risk", f"{probability*100:.2f}%")
    else:
        st.success('✅ Employee is likely to stay in the company.')
        st.metric("Attrition Risk", f"{probability*100:.2f}%")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Built using Python, Scikit-learn, Streamlit, SHAP, and Power BI")