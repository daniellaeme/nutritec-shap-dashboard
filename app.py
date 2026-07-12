import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

from PIL import Image
import cv2

import shap
import matplotlib.pyplot as plt
import google.genai as genai


st.set_page_config(page_title="NutritecAI AI-Integrated Diagnostic Dashboard")
st.title("NutritecAI AI-Integrated Diagnostic Dashboard")
st.caption('')


@st.cache_resource
def load_model(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.dirname(current_dir)
    model_dir = os.path.join(current_dir, 'models')

    filepath = os.path.join(model_dir, filename)
    rfc_model = joblib.load(filepath)

    return rfc_model


def translate_image(picture):
    # Image Translation: Pillow to Numpy and OpenCV, RGB to HSV
    img = Image.open(picture).convert('RGB')
    img_array = np.array(img)
    hsv_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)

    # Create a mask for Red
    # Red wraps around the HSV spectrum (ends of 0 and 180)
    # OpenCV HSV ranges: H (0-180), S (0-255), V (0-255)

    # Mask 1: For the low end of the red spectrum (0 to 10 degrees)
    # Increased S from 50 to 150 to ignore pale skin tones
    # Increased V from 50 to 70 to ignore dark ambient shadows
    lower_red1 = np.array([0, 150, 70])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)

    # Mask 2: For the high end of the red spectrum (170 to 180 degrees)
    lower_red2 = np.array([170, 50, 70])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)

    full_red_mask = mask1 + mask2

    # Calculate Clinical Value
    # Extract only the saturation values (channel 1) where the mask is active
    saturation_channel = hsv_img[:, :, 1]
    red_pixels = saturation_channel[full_red_mask > 0]

    # Handle edge case where no red color is found in the webcam snapshot
    if len(red_pixels) == 0:
        raw_score = 0.0
        st.warning("No red color detected. Please ensure your red circle is visible.")
    else:
        raw_score = np.mean(red_pixels)

    return raw_score


model = load_model('rfc_model.pkl')


# Input
st.subheader('Patient Demographics')

col1, col2 = st.columns(2)
with col1:
    age = st.slider('Age', min_value=25, max_value=55, value=40)
with col2:
    bmi = st.number_input('BMI', min_value=18.0, max_value=24.0, value=21.5)
family_history = st.selectbox('Do you have a family history of breast cancer?', ('Yes', 'No') )
# mammaglobin_level = st.slider('Mammaglobin Level', min_value=0.0, max_value=10.0, value=5.0)


st.subheader('Diagnostic Input')
st.write('Hold your mock red strip up to the camera and take a picture.')
# enable = st.checkbox('Enable camera')
picture = st.camera_input('Take a picture') #, disabled=enable)

if not picture:
    st.error('Please take a picture of your test strip to proceed')

elif picture:
    st.image(picture, caption='Captured Image')

    raw_score = translate_image(picture)

    # Mathematical Conversion
    # Scale from 0-255 range to 0.0-10.0 Mammaglobin range
    mammaglobin_level = (raw_score / 255.0) * 10.0
    mammaglobin_level = round(mammaglobin_level, 2)

    st.info(f"Raw Red Intensity Score: {round(raw_score, 2)} / 255")
    st.success(f"Calculated Mammaglobin Level: {mammaglobin_level} ng/mL (Passed to Model)")


    # Perform Analysis
    if st.button('Analyse Risk Profile'):
        if family_history == 'Yes':
            family_history_num = 1
        else:
            family_history_num = 0

        input_data = pd.DataFrame({
            'Age': [age],
            'BMI': [bmi],
            'Family_History': [family_history_num],
            'Saliva_Mammaglobin': [mammaglobin_level]
        })

        # Perform Prediction
        probabilities = model.predict_proba(input_data)
        at_risk_proba = probabilities[0][1]
        at_risk_proba = round(at_risk_proba * 100, 2)


        # Display Results
        st.subheader('Risk Profile')
        st.metric('Risk Probability', value=str(at_risk_proba) + '%')

        if at_risk_proba < 30:
            st.success('Risk level is within normal parameters.')
        elif 30 < at_risk_proba < 60:
            st.warning('Elevated risk detected. Further consultation recommended.')
        else:
            st.error('High risk detected based on diagnostic markers.')

        # SHAP Explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values[:, :, 1], input_data, plot_type='bar', show=False)
        plt.xlabel('Mean SHAP value')
        plt.title('SHAP Summary Plot')
        fig.tight_layout()
        with st.expander('View AI Decision Breakdown', expanded=False):
            st.pyplot(fig)

        # GenAI
        prompt = (f'You are a clinical assistant. A patient (Age:{age}, BMI: {bmi}, '
                  f'Family History: {family_history}) was tested. The optical scanner detected a '
                  f'raw mammaglobin level of {mammaglobin_level}. Our Random Forest model predicts a {at_risk_proba}% '
                  f'risk of anomaly. Based on this data, write a brief, objective, 3-sentence clinical summary for '
                  f'the attending physician. Do not provide a medical diagnosis.')

        try:
            client = genai.Client(api_key=st.secrets['GEMINI_API_KEY'])
            response = client.models.generate_content(
                model='gemini-3.5-flash',
                contents=prompt
            )
            st.subheader('AI Clinical Summary')
            st.info(response.text)
        except:
            st.warning('AI Clinical summary unavailable at this time')
