# Multimodal AI Diagnostic Dashboard (Nutritec AI Integration Concept)
Nutritec AI relies on a saliva test strip that likely changes color based on mammaglobin concentration. 
This project is a streamlit web app that uses the device camera to read a simulated physical test strip 
(Computer Vison), predicts the cancer risk (Machine Learning), explains the math (SHAP) and generates a human-readable 
clinical summary (Generative-AI). 

## Problem
NutritecAI uses physical test strips for testing. To be trusted by physicians, the software must be seamless and 
transparent.

## Solution
This solution acts as a physical to digital bridge (with the camera) and a Trust Layer (with SHAP and Gemini).

## Tech Stack
### - Machine Learning (Scikit-Learn Random Forest)
Random Forest was chosen because human biology is not linear. The model prevents overfittimg and outputs a 
probability rather than a rigid Yes/No, keeping the doctor in the loop.
### - Computer Vision (OpenCV)
HSV color masking was chosen instead of RGB because HSV is much better at ignoring ambient room lighting/shadows 
when reading a physical test strip via webcam.
### - Explainability (SHAP)
SHAP uses game theory to prove mathematically exactly which biological feature triggered the high-risk alert.
### - Generative AI (Google Gemini)
The LLM acts as an automated medical scribe, translating raw SHAP arrays into a readable, professional 3-sentence 
summary.

## Data Engineering
As Nutritec's hardware is proprietary,  a standard public dataset could not be used so, NumPy distributions were 
used to synthesize mock data. I intentionally created overlapping standard deviations between the "Healthy" and 
"At-Risk" mammaglobin levels to mimic real-world biological noise and force the ML model to evaluate the context 
(Age, BMI) rather than just memorizing a threshold.

## Setup & Installation
Clone the repository
'''
git clone https://github.com/daniellaeme/nutritec-shap-dashboard.git
'''
Install the required packages
'''
pip install -r requirements.txt
'''
Create a .streamlit/secrets.toml
Run the streamlit app
'''
streamlit run app.py
'''

### Future Scalability
- Swap Streamlit UI for e.g. React.
- Set up proper database server e.g. Celery for patient records.

### _Project Status: Completed_
- [x] Machine Learning
- [x] SHAP
- [x] Computer Vision
- [X] Generative AI
