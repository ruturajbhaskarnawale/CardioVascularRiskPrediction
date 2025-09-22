import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from math import radians, sin, cos, sqrt, atan2
import datetime
from fpdf import FPDF
import base64
import os
import database as db 

# --- Load Model and Scaler ---
try:
    best_model = joblib.load('cardio_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Error: Model or scaler files not found.")
    st.error("Please ensure 'cardio_model.pkl' and 'scaler.pkl' are in the same directory as this script.")
    st.stop()

numerical_cols = ['age', 'height', 'ap_hi', 'ap_lo', 'bmi']

def get_image_as_base64(path):
    """Reads an image file and returns it as a Base64 encoded string."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as img_file:
        # Use a f-string to embed the encoded data in the correct data URI format
        return f"data:image/jpeg;base64,{base64.b64encode(img_file.read()).decode()}"
from urllib.parse import urlparse, parse_qs

def get_youtube_id(url):
    """Extracts the YouTube video ID from a URL."""
    if not url:
        return None
    try:
        if "youtu.be" in url:
            return url.split("/")[-1].split("?")[0]
        if "youtube.com" in url:
            query = urlparse(url).query
            params = parse_qs(query)
            return params.get("v", [None])[0]
    except (IndexError, KeyError):
        return None
    return None
# --- Mock Model for Comparison (for demonstration purposes) ---
# This simulates a second model without needing to load another file.
def mock_predict_disease(user_input):
    """
    A mock model for comparison.
    """
    bmi = user_input['weight'] / (user_input['height'] / 100)**2
    ap_hi_score = user_input['ap_hi'] * 0.005
    bmi_score = bmi * 0.015
    age_score = user_input['age'] * 0.01

    risk_score = (ap_hi_score + bmi_score + age_score)
    prob = max(0.01, min(0.99, risk_score / 2.0)) # Ensure probability is between 0 and 1
    
    prediction = 1 if prob > 0.5 else 0
    result = "Cardiovascular Disease" if prediction == 1 else "No Cardiovascular Disease"
    
    return result, prob

# --- Prediction Function (for single input) ---
def predict_disease(user_input):
    """
    Prepares user input, scales numerical features, and predicts cardiovascular disease risk.
    """
    bmi = user_input['weight'] / (user_input['height'] / 100)**2
    age_in_days = user_input['age'] * 365.25

    data_for_prediction = {
        'age': age_in_days,
        'gender': 1 if user_input['gender'] == 'Female' else 0,
        'height': user_input['height'],
        'ap_hi': user_input['ap_hi'],
        'ap_lo': user_input['ap_lo'],
        'cholesterol': user_input['cholesterol'],
        'gluc': user_input['gluc'],
        'smoke': user_input['smoke'],
        'alco': user_input['alco'],
        'active': user_input['active'],
        'bmi': bmi
    }

    model_features = [
        'age', 'gender', 'height', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
        'smoke', 'alco', 'active', 'bmi'
    ]

    input_df = pd.DataFrame([data_for_prediction], columns=model_features)
    df_scaled = input_df.copy()
    df_scaled[numerical_cols] = scaler.transform(df_scaled[numerical_cols])

    prob = best_model.predict_proba(df_scaled)[:, 1][0]
    prediction = best_model.predict(df_scaled)[0]
    result = "Cardiovascular Disease" if prediction == 1 else "No Cardiovascular Disease"

    return result, prob

# --- Bulk Prediction Function ---
def bulk_predict_disease(df_bulk):
    """
    Processes a DataFrame of user inputs for bulk prediction.
    """
    df_processed = df_bulk.copy()

    gender_map_csv = {"Male": 0, "Female": 1, 0:0, 1:1}
    cholesterol_map_csv = {"Normal (1)": 1, "Above Normal (2)": 2, "Well Above (3)": 3, 1:1, 2:2, 3:3}
    gluc_map_csv = {"Normal (1)": 1, "Above Normal (2)": 2, "Well Above (3)": 3, 1:1, 2:2, 3:3}
    smoke_map_csv = {"Non-smoker": 0, "Smoker": 1, 0:0, 1:1}
    alco_map_csv = {"Non-drinker": 0, "Moderate Drinker": 1, "Heavy Drinker": 2, 0:0, 1:1, 2:2}
    active_map_csv = {"Sedentary": 0, "Moderately Active": 1, "Very Active": 2, 0:0, 1:1, 2:2}

    try:
        required_cols = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        if not all(col in df_processed.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_processed.columns]
            raise KeyError(f"Missing required columns in CSV: {', '.join(missing)}")

        df_processed['bmi'] = df_processed['weight'] / (df_processed['height'] / 100)**2
        df_processed['age'] = df_processed['age'] * 365.25

        df_processed['gender'] = df_processed['gender'].map(gender_map_csv).fillna(0)
        df_processed['cholesterol'] = df_processed['cholesterol'].map(cholesterol_map_csv).fillna(1)
        df_processed['gluc'] = df_processed['gluc'].map(gluc_map_csv).fillna(1)
        df_processed['smoke'] = df_processed['smoke'].map(smoke_map_csv).fillna(0)
        df_processed['alco'] = df_processed['alco'].map(alco_map_csv).fillna(0)
        df_processed['active'] = df_processed['active'].map(active_map_csv).fillna(1)

        columns_to_drop = ['weight']
        if 'id' in df_processed.columns:
            columns_to_drop.append('id')
        if 'stress' in df_processed.columns:
            columns_to_drop.append('stress')

        df_processed.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')

        model_features_order = [
            'age', 'gender', 'height', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
            'smoke', 'alco', 'active', 'bmi'
        ]
        df_processed = df_processed[model_features_order]

        df_processed[numerical_cols] = scaler.transform(df_processed[numerical_cols])

        probabilities = best_model.predict_proba(df_processed)[:, 1]
        predictions = best_model.predict(df_processed)

        df_bulk['Predicted_Cardio_Disease'] = np.where(predictions == 1, 'Yes', 'No')
        df_bulk['Prediction_Probability'] = probabilities
        return df_bulk

    except KeyError as e:
        st.error(f"Error: Missing or incorrect column in CSV: {e}. Please ensure your CSV has all necessary columns: age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active.")
        return None
    except ValueError as e:
        st.error(f"Error with data types or values in CSV: {e}. Please check that numerical columns contain numbers and categorical columns contain expected values.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during bulk prediction: {e}. Please check your CSV file format and data.")
        return None

# --- NEW AND UPDATED: Function to generate EDA plots for bulk data ---
def generate_bulk_eda_plots(df_bulk):
    """
    Generates and returns a list of impressive Plotly plots for advanced EDA of bulk data.
    """
    plots = []
    
    df_processed_for_plot = df_bulk.copy()
    
    # Define mappings for human-readable labels
    gender_map = {0: 'Male', 1: 'Female', 'Male': 'Male', 'Female': 'Female'}
    cholesterol_map = {1: 'Normal', 2: 'Above Normal', 3: 'Well Above', 'Normal (1)': 'Normal', 'Above Normal (2)': 'Above Normal', 'Well Above (3)': 'Well Above'}
    gluc_map = {1: 'Normal', 2: 'Above Normal', 3: 'Well Above', 'Normal (1)': 'Normal', 'Above Normal (2)': 'Above Normal', 'Well Above (3)': 'Well Above'}
    smoke_map = {0: 'Non-smoker', 1: 'Smoker', 'Non-smoker': 'Non-smoker', 'Smoker': 'Smoker'}
    alco_map = {0: 'Non-drinker', 1: 'Moderate Drinker', 2: 'Heavy Drinker', 'Non-drinker': 'Non-drinker', 'Moderate Drinker': 'Moderate Drinker', 'Heavy Drinker': 'Heavy Drinker'}
    active_map = {0: 'Sedentary', 1: 'Moderately Active', 2: 'Very Active', 'Sedentary': 'Sedentary', 'Moderately Active': 'Moderately Active', 'Very Active': 'Very Active'}

    # Apply mappings
    for col, mapping in [('gender', gender_map), ('cholesterol', cholesterol_map), ('gluc', gluc_map),
                          ('smoke', smoke_map), ('alco', alco_map), ('active', active_map)]:
        if col in df_processed_for_plot.columns:
            df_processed_for_plot[col] = df_processed_for_plot[col].map(mapping).fillna(df_processed_for_plot[col])

    # Calculate BMI for plotting
    if 'height' in df_processed_for_plot.columns and 'weight' in df_processed_for_plot.columns:
        df_processed_for_plot['bmi'] = df_processed_for_plot['weight'] / (df_processed_for_plot['height'] / 100)**2

    # --- 1. Box Plots for Numerical Data ---
    numerical_columns = ['age', 'bmi', 'ap_hi', 'ap_lo']
    for col in numerical_columns:
        if col in df_processed_for_plot.columns:
            fig = px.box(
                df_processed_for_plot,
                y=col,
                title=f'Distribution of {col.title()}',
                color_discrete_sequence=['#4B90F9']
            )
            fig.update_layout(
                plot_bgcolor='#222E3A',
                paper_bgcolor='#222E3A',
                font_color='#E0E0E0',
                title_font_color='#E0E0E0'
            )
            plots.append(fig)

    # --- 2. Sunburst Chart for Categorical Data ---
    categorical_columns = ['gender', 'smoke', 'cholesterol', 'gluc']
    if all(col in df_processed_for_plot.columns for col in categorical_columns[:2]):
        fig_sunburst = px.sunburst(
            df_processed_for_plot,
            path=['gender', 'smoke'],
            title='Patient Demographics by Gender and Smoking Status'
        )
        fig_sunburst.update_layout(
            plot_bgcolor='#222E3A',
            paper_bgcolor='#222E3A',
            font_color='#E0E0E0',
            title_font_color='#E0E0E0'
        )
        plots.append(fig_sunburst)
    
    # --- 3. Violin Plot for Numerical vs. Categorical Comparison ---
    if all(col in df_processed_for_plot.columns for col in ['age', 'cholesterol']):
        fig_violin = px.violin(
            df_processed_for_plot,
            y='age',
            x='cholesterol',
            title='Age Distribution by Cholesterol Level',
            box=True,
            points='all',
            color='cholesterol'
        )
        fig_violin.update_layout(
            plot_bgcolor='#222E3A',
            paper_bgcolor='#222E3A',
            font_color='#E0E0E0',
            title_font_color='#E0E0E0'
        )
        plots.append(fig_violin)

    return plots

# --- Recommendation Generation Function (remains the same) ---
def generate_recommendations(user_input, prob, result):
    """
    Generates personalized health recommendations based on prediction results and user inputs.
    """
    rec = []
    factors = []

    bmi = user_input['weight'] / (user_input['height'] / 100)**2

    if user_input['age'] >= 50:
        factors.append(f"• Age: {int(user_input['age'])} years (increased risk over 50)")
    if bmi >= 25:
        factors.append(f"• BMI: {bmi:.1f} (overweight - consider weight management)")
    if user_input['ap_hi'] >= 130 or user_input['ap_lo'] >= 85:
        factors.append(f"• Blood Pressure: {user_input['ap_hi']}/{user_input['ap_lo']} mmHg (elevated)")
    if user_input['cholesterol'] > 1:
        chol_text = "Above Normal" if user_input['cholesterol'] == 2 else "Well Above Normal"
        factors.append(f"• Cholesterol: {chol_text}")
    if user_input['gluc'] > 1:
        gluc_text = "Above Normal" if user_input['gluc'] == 2 else "Well Above Normal"
        factors.append(f"• Glucose: {gluc_text}")
    if user_input['smoke'] == 1:
        factors.append("• Smoking: Major risk factor")
    if user_input['alco'] > 0:
        alco_text = "Moderate consumption" if user_input['alco'] == 1 else "Heavy consumption"
        factors.append(f"• Alcohol: {alco_text} increases risk")
    if user_input['active'] == 0:
        factors.append("• Physical Activity: Sedentary lifestyle")
    if user_input['stress'] == 2:
        factors.append("• Stress: High stress levels impact cardiovascular health")

    if not factors:
        factors.append("• No significant risk factors identified based on inputs.")

    if prob > 0.75:
        risk_level = "High Risk"
        risk_color = "red"
    elif prob > 0.45:
        risk_level = "Moderate Risk"
        risk_color = "orange"
    else:
        risk_level = "Low Risk"
        risk_color = "green"

    if "Cardiovascular Disease" in result:
        rec.append("🚨 **IMMEDIATE ACTIONS:**")
        rec.append("• Consult a cardiologist within 2 weeks for a comprehensive evaluation.")
        rec.append("• Request comprehensive blood work (lipid panel, HbA1c, CRP).")
        rec.append("• Begin daily blood pressure monitoring at home.")

        rec.append("\n💊 **MEDICATION CONSIDERATIONS:**")
        rec.append("• Statins may be recommended based on cholesterol levels and overall risk.")
        rec.append("• ACE inhibitors or ARBs for blood pressure management, if elevated.")
        rec.append("• Low-dose aspirin therapy (only if prescribed by a doctor).")

        rec.append("\n🏃 **LIFESTYLE MODIFICATIONS:**")
        if bmi > 25:
            rec.append(f"• Target weight loss: Approximately {(bmi-25)*2:.1f} kg to reach a healthy BMI.")
        else:
            rec.append("• Maintain current healthy weight.")
        rec.append("• Adopt the DASH diet (Dietary Approaches to Stop Hypertension) focusing on fruits, vegetables, and whole grains.")
        rec.append("• Aim for at least 150 minutes/week of moderate aerobic exercise, plus strength training 2-3 times/week.")

        if user_input['smoke'] == 1:
            rec.append("\n🚭 **SMOKING CESSATION:**")
            rec.append("• Enroll in a smoking cessation program immediately.")
            rec.append("• Consider nicotine replacement therapy or medication as advised by your doctor.")
            rec.append("• Set a quit date within 7 days and seek support.")

        rec.append("\n📊 **RISK MANAGEMENT & MONITORING:**")
        rec.append("• Work towards A1C < 7.0%, LDL < 70 mg/dL, and BP < 130/80 mmHg targets.")
        rec.append("• Consider a cardiac stress test within 1 month, as recommended by your cardiologist.")
        rec.append("• Discuss a sleep study if symptoms of sleep apnea are present.")
    else:
        rec.append("✅ **PREVENTIVE STRATEGIES:**")
        rec.append("• Continue to maintain current healthy habits and positive lifestyle choices.")
        rec.append("• Schedule an annual cardiovascular risk assessment with your healthcare provider.")

        rec.append("\n🥗 **NUTRITION OPTIMIZATION:**")
        rec.append("• Adopt a Mediterranean diet pattern (rich in fruits, vegetables, whole grains, healthy fats).")
        rec.append("• Limit sodium intake to less than 1500mg/day and saturated fat to less than 7% of daily calories.")
        rec.append("• Increase intake of omega-3 fatty acids (e.g., fatty fish, flaxseeds, chia seeds).")

        rec.append("\n💪 **FITNESS PLAN:**")
        if user_input['active'] == 0:
            rec.append("• Start with 30 minutes/day of brisk walking, 5 days/week.")
            rec.append("• Gradually increase to 150 minutes/week of moderate-intensity aerobic activity.")
        elif user_input['active'] == 1:
            rec.append("• Maintain at least 150 minutes/week of moderate activity.")
            rec.append("• Add 2-3 days/week of strength training exercises.")
        else:
            rec.append("• Continue your current activity level.")
            rec.append("• Consider incorporating High-Intensity Interval Training (HIIT) 1-2 times/week.")

        rec.append("\n🛡 **RISK REDUCTION:**")
        if user_input['cholesterol'] > 1:
            rec.append(f"• Focus on reducing LDL cholesterol through dietary changes (e.g., soluble fiber, plant sterols).")
        if bmi > 25:
            rec.append(f"• Target 5-10% weight loss to significantly improve cardiovascular health.")
        if user_input['stress'] == 2:
            rec.append("• Implement daily stress-reduction techniques such as mindfulness, meditation, or yoga.")

    screening = []
    if user_input['age'] < 40:
        screening.append("• Lipid panel every 4-5 years.")
        screening.append("• Blood pressure check annually.")
    elif user_input['age'] < 50:
        screening.append("• Lipid panel every 2-3 years.")
        screening.append("• Blood glucose every 3 years.")
        screening.append("• Blood pressure check annually.")
    else:
        screening.append("• Comprehensive metabolic panel annually.")
        screening.append("• Consider cardiac calcium scoring, as advised by your doctor.")
        screening.append("• Blood pressure check annually.")

    if user_input['cholesterol'] > 1 or user_input['gluc'] > 1:
        screening.append("• Semi-annual lipid and/or glucose monitoring.")

    resources = []
    if "Cardiovascular Disease" in result:
        resources.append("• American Heart Association: [heart.org](https://www.heart.org)")
        resources.append("• CardioSmart Patient Education: [cardiosmart.org](https://www.cardiosmart.org)")
        resources.append("• Million Hearts Initiative: [millionhearts.hhs.gov](https://millionhearts.hhs.gov)")
    else:
        resources.append("• CDC Heart Disease Prevention: [cdc.com/heartdisease](https://www.cdc.com/heartdisease)")
        resources.append("• DASH Diet Resources: [nhlbi.nih.gov/education/dash-eating-plan](https://www.nhlbi.nih.gov/education/dash-eating-plan)")
        resources.append("• American College of Cardiology Prevention Tools: [acc.org](https://www.acc.org)")

    return {
        "risk_level": risk_level,
        "risk_color": risk_color,
        "probability": f"Risk Probability: {prob*100:.1f}%",
        "prob_value": prob,
        "factors": "\n".join(factors),
        "recommendations": "\n".join(rec),
        "screening": "\n".join(screening) if screening else "• Annual physical with basic blood work.",
        "resources": "\n".join(resources)
    }

class PDF(FPDF):
    def header(self):
        # Logo could be added here
        self.set_font('DejaVu', 'B', 16)
        self.cell(0, 10, 'CardioHealth Risk Report', 0, 1, 'C')
        self.set_font('DejaVu', 'I', 10)
        self.cell(0, 8, 'Personalized Cardiovascular Assessment', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVu', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'L')
        self.cell(0, 10, datetime.date.today().strftime("%B %d, %Y"), 0, 0, 'C')
        self.cell(0, 10, 'CardioHealth Risk Predictor Pro', 0, 0, 'R')

    def chapter_title(self, title):
        self.set_font('DejaVu', 'B', 14)
        self.set_fill_color(230, 230, 250) # A light blue/lavender background
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('DejaVu', '', 11)
        self.multi_cell(0, 6, body)
        self.ln()

    def result_summary(self, rec_data):
        self.set_font('DejaVu', 'B', 24)
        if rec_data['risk_color'] == 'red':
            self.set_text_color(220, 50, 50)
        elif rec_data['risk_color'] == 'orange':
            self.set_text_color(255, 165, 0)
        else:
            self.set_text_color(34, 139, 34)
        
        self.cell(0, 12, f"Result: {rec_data['risk_level']}", 0, 1, 'C')
        self.set_font('DejaVu', '', 16)
        self.set_text_color(0, 0, 0) # Reset color
        self.cell(0, 10, f"({rec_data['probability']})", 0, 1, 'C')
        self.ln(10)

def generate_pdf_report(user_input_report, rec_data):
    """
    Generates a downloadable PDF report from user inputs and prediction results.
    """
    pdf = PDF()
    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    pdf.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)
    pdf.add_font('DejaVu', 'I', 'DejaVuSans-Oblique.ttf', uni=True)
    pdf.add_font('DejaVu', 'BI', 'DejaVuSans-BoldOblique.ttf', uni=True)
    pdf.add_page()
    
    # 1. Prediction Summary
    pdf.result_summary(rec_data)

    # 2. Key Risk Factors
    pdf.chapter_title('Key Risk Factors Identified')
    pdf.chapter_body(rec_data['factors'])

    # 3. Personalized Recommendations
    pdf.chapter_title('Personalized Recommendations & Actions')
    pdf.chapter_body(rec_data['recommendations'])

    # 4. Screening & Monitoring Plan
    pdf.chapter_title('Screening & Monitoring Plan')
    pdf.chapter_body(rec_data['screening'])

    # 5. User Input Summary
    pdf.add_page()
    pdf.chapter_title('Patient Data Provided for this Assessment')
    
    # Create a simple table for user inputs
    pdf.set_font('DejaVu', 'B', 11)
    col_width = pdf.w / 2.5
    row_height = 8

    # Reverse map the categorical values for readability
    cholesterol_rev_map = {1: "Normal", 2: "Above Normal", 3: "Well Above"}
    gluc_rev_map = {1: "Normal", 2: "Above Normal", 3: "Well Above"}
    smoke_rev_map = {0: "Non-smoker", 1: "Smoker"}
    alco_rev_map = {0: "Non-drinker", 1: "Moderate", 2: "Heavy"}
    active_rev_map = {0: "Sedentary", 1: "Moderately Active", 2: "Very Active"}
    stress_rev_map = {0: "Low", 1: "Moderate", 2: "High"}

    input_data_table = {
        "Patient Full Name": user_input_report.get('full_name', 'N/A'),
        "Patient Phone": user_input_report.get('phone_number', 'N/A'),
        "Age": f"{int(user_input_report['age'])} years",
        "Gender": user_input_report['gender'],
        "Height": f"{int(user_input_report['height'])} cm",
        "Weight": f"{user_input_report['weight']:.1f} kg",
        "Systolic BP": f"{int(user_input_report['ap_hi'])} mmHg",
        "Diastolic BP": f"{int(user_input_report['ap_lo'])} mmHg",
        "Cholesterol": cholesterol_rev_map.get(user_input_report['cholesterol'], "N/A"),
        "Glucose": gluc_rev_map.get(user_input_report.get('gluc', 1), "N/A"),
        "Smoker": smoke_rev_map.get(user_input_report['smoke'], "N/A"),
        "Alcohol Intake": alco_rev_map.get(user_input_report.get('alco', 0), "N/A"),
        "Activity Level": active_rev_map.get(user_input_report.get('active', 1), "N/A"),
        "Stress Level": stress_rev_map.get(user_input_report.get('stress', 1), "N/A"),
    }
    
    for key, value in input_data_table.items():
        pdf.set_font('DejaVu', 'B', 11)
        pdf.cell(col_width, row_height, f"{key}:", border=1)
        pdf.set_font('DejaVu', '', 11)
        pdf.cell(col_width, row_height, str(value), border=1)
        pdf.ln(row_height)
    
    pdf.ln(10)
    
    # Disclaimer
    pdf.set_font('DejaVu', 'I', 9)
    pdf.multi_cell(0, 5, "Disclaimer: This assessment is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any health concerns.")

    # Return the PDF data as bytes
    return bytes(pdf.output(dest='S'))

# --- Educational Content Hub Data ---
EDUCATIONAL_CONTENT = {
    "Understanding Blood Pressure": {
        "video_url": "https://youtu.be/0gnJk_vjtWY?si=CZsYI8wQNwlPRr-X", # Verified Link
        "summary": "Blood pressure is the force of blood pushing against the walls of your arteries. It's recorded as two numbers: systolic (the higher number) and diastolic (the lower number). High blood pressure, or hypertension, can damage your arteries and lead to serious health problems like heart disease and stroke.",
        "key_points": [
            "**Systolic Pressure**: Measures pressure in your arteries when your heart beats.",
            "**Diastolic Pressure**: Measures pressure in your arteries when your heart rests between beats.",
            "**Normal BP**: Less than 120/80 mmHg.",
            "**Elevated BP**: 120-129 systolic and less than 80 diastolic.",
            "**Hypertension Stage 1**: 130-139 systolic or 80-89 diastolic."
        ],
        "image": r"C:\Users\rutur\OneDrive\Documents\5TH_ML\educationalHub_image\bloodPressure.jpg" # Verified Link
    },
    "Managing Cholesterol": {
        "video_url": "https://www.youtube.com/watch?v=fLonh7ZesKs", # Verified Link
        "summary": "Cholesterol is a waxy substance found in your blood. While your body needs it to build healthy cells, high levels of cholesterol can increase your risk of heart disease. It can lead to fatty deposits in your blood vessels, which can eventually block blood flow.",
        "key_points": [
            "**LDL (Low-Density Lipoprotein)**: Known as 'bad' cholesterol. High levels lead to plaque buildup in arteries.",
            "**HDL (High-Density Lipoprotein)**: Known as 'good' cholesterol. It helps carry away LDL cholesterol.",
            "**Diet**: Reduce saturated and trans fats. Increase soluble fiber (oats, fruits, beans).",
            "**Exercise**: Regular physical activity can raise HDL and lower LDL."
        ],
        "image": r"C:\Users\rutur\OneDrive\Documents\5TH_ML\educationalHub_image\cholestrol.jpg" # Verified Link
    },
    "The Importance of Physical Activity": {
        "video_url": "https://www.youtube.com/watch?v=c0R7z88Dk7E", # Verified Link
        "summary": "Regular physical activity is one of the most important things you can do for your heart health. It helps control weight, reduce blood pressure and cholesterol, and lower your risk of heart disease. Aim for a mix of aerobic and strength training exercises.",
        "key_points": [
            "**Aerobic Exercise**: Aim for at least 150 minutes of moderate-intensity activity (like brisk walking) or 75 minutes of vigorous activity (like running) per week.",
            "**Strength Training**: Include muscle-strengthening activities at least two days per week.",
            "**Consistency is Key**: Even short bursts of activity throughout the day add up.",
            "**Benefits**: Improves circulation, strengthens the heart muscle, and helps manage stress."
        ],
        "image": r"C:\Users\rutur\OneDrive\Documents\5TH_ML\educationalHub_image\physicalActivity.jpg" # Verified Link
    },
    "The DASH Diet for a Healthy Heart": {
        "video_url": "https://www.youtube.com/watch?v=jaln_gM_0_Y", # Verified Link
        "summary": "The DASH (Dietary Approaches to Stop Hypertension) diet is a lifelong approach to healthy eating that's designed to help treat or prevent high blood pressure. It encourages you to reduce the sodium in your diet and eat a variety of foods rich in nutrients that help lower blood pressure, such as potassium, calcium, and magnesium.",
        "key_points": [
            "**Focus On**: Vegetables, fruits, and whole grains.",
            "**Include**: Fat-free or low-fat dairy products, fish, poultry, beans, nuts, and vegetable oils.",
            "**Limit**: Foods high in saturated fat (fatty meats, full-fat dairy), sugar-sweetened beverages, and sweets.",
            "**Reduce Sodium**: Aim for 2,300 mg per day, with an ideal limit of 1,500 mg for most adults."
        ],
        "image": r"C:\Users\rutur\OneDrive\Documents\5TH_ML\educationalHub_image\dashDiet.jpg" # Verified Link
    },
    "How Stress Affects Your Heart": {
        "video_url": "https://www.youtube.com/watch?v=00j_Am_K-sU", # Verified Link
        "summary": "Stress can lead to behaviors and factors that increase heart disease risk, such as high blood pressure, high cholesterol, smoking, physical inactivity, and overeating. Your body's response to stress is to release adrenaline, which can cause your breathing and heart rate to speed up and your blood pressure to rise. Chronic stress can be very damaging to your heart over time.",
        "key_points": [
            "**Indirect Effects**: Stress can lead to poor lifestyle choices like unhealthy diet, smoking, or lack of exercise.",
            "**Direct Effects**: Chronic stress can lead to high blood pressure and may damage artery walls.",
            "**Management Techniques**: Practice relaxation techniques like meditation, deep breathing, or yoga.",
            "**Stay Active**: Physical activity is a powerful natural stress reliever."
        ],
        "image": r"C:\Users\rutur\OneDrive\Documents\5TH_ML\educationalHub_image\streess.jpg" # Verified Link
    },
    "The Link Between Diabetes and Heart Disease": {
        "video_url": "https://www.youtube.com/watch?v=UK43o4_y3yA", # Verified Link
        "summary": "People with diabetes are at a much higher risk of developing heart disease. High blood sugar from diabetes can damage blood vessels and the nerves that control your heart. Over time, this damage can lead to conditions like high blood pressure and high cholesterol, which are major risk factors for heart attacks and strokes.",
        "key_points": [
            "**Higher Risk**: Adults with diabetes are nearly twice as likely to die from heart disease or stroke as people without diabetes.",
            "**Manage Your ABCs**: **A**1C (blood sugar), **B**lood pressure, and **C**holesterol.",
            "**Lifestyle is Key**: A heart-healthy diet and regular exercise are critical for managing both diabetes and heart health.",
            "**Don't Smoke**: Smoking significantly increases the risk of heart disease, especially for people with diabetes."
        ],
        "image": r"C:\Users\rutur\OneDrive\Documents\5TH_ML\educationalHub_image\diabeties.jpg" # Verified Link
    }
}


# --- Simulated Geospatial Data and Function (remains the same) ---
# CITY_COORDINATES = {
#     "Mumbai": {"lat": 19.0760, "lon": 72.8777},
#     "New Delhi": {"lat": 28.6139, "lon": 77.2090},
#     "Bengaluru": {"lat": 12.9716, "lon": 77.5946},
#     "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
#     "Chennai": {"lat": 13.0827, "lon": 80.2707},
#     "Kolkata": {"lat": 22.5726, "lon": 88.3639},
#     "London": {"lat": 51.5074, "lon": -0.1278},
#     "New York": {"lat": 40.7128, "lon": -74.0060},
#     "Tokyo": {"lat": 35.6895, "lon": 139.6917},
#     "Sydney": {"lat": -33.8688, "lon": 151.2093},
#     "Toronto": {"lat": 43.6532, "lon": -79.3832},
#     "Berlin": {"lat": 52.5200, "lon": 13.4050},
#     "Paris": {"lat": 48.8566, "lon": 2.3522},
#     "Anytown": {"lat": 34.0522, "lon": -118.2437},
# }
MAJOR_CITIES = {
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Pune": {"lat": 18.5204, "lon": 73.8567},
    "Nagpur": {"lat": 21.1458, "lon": 79.0882},
    "Nashik": {"lat": 20.0112, "lon": 73.7909},
    "Aurangabad": {"lat": 19.8762, "lon": 75.3433}
}
SIMULATED_RESOURCES = {
    "cardiologist": [
        {"name": "Mumbai Heart Clinic", "address": "123 Health St, Mumbai", "lat": 19.0780, "lon": 72.8800, "description": "Leading cardiac care with advanced diagnostics."},
        {"name": "Mumbai Cardio Specialists", "address": "Bandra West, Mumbai", "lat": 19.0590, "lon": 72.8290, "description": "Expert cardiologists for complex heart conditions."},
        {"name": "New Delhi Heart Institute", "address": "456 Wellness Ave, New Delhi", "lat": 28.6150, "lon": 77.2120, "description": "Comprehensive heart health services."},
        {"name": "Bengaluru Cardiac Center", "address": "789 Life Rd, Bengaluru", "lat": 12.9730, "lon": 77.5960, "description": "Specialized in interventional cardiology."},
        {"name": "Hyderabad Heart Care", "address": "Gachibowli, Hyderabad", "lat": 17.4400, "lon": 78.3900, "description": "Modern facility for heart surgeries and consultations."},
        {"name": "Chennai Cardio Expert", "address": "Anna Salai, Chennai", "lat": 13.0600, "lon": 80.2700, "description": "Renowned for preventive cardiology."},
        {"name": "Kolkata Heart Clinic", "address": "Park Street, Kolkata", "lat": 22.5400, "lon": 88.3500, "description": "Dedicated to patient-centric heart care."},
        {"name": "London Cardiac Health", "address": "10 Downing St, London", "lat": 51.5080, "lon": -0.1300, "description": "NHS and private cardiac services."},
        {"name": "New York Heart Center", "address": "Broadway, New York", "lat": 40.7150, "lon": -74.0080, "description": "Top-rated cardiac surgeons and specialists."},
        {"name": "Tokyo Cardiovascular Clinic", "address": "Shinjuku, Tokyo", "lat": 35.6900, "lon": 139.7000, "description": "Advanced diagnostics and treatment for heart disease."},
        {"name": "Sydney Heart & Vascular", "address": "Circular Quay, Sydney", "lat": -33.8600, "lon": 151.2100, "description": "Leading experts in vascular and heart health."},
        {"name": "Toronto Cardiac Care", "address": "Bay Street, Toronto", "lat": 43.6500, "lon": -79.3800, "description": "Holistic approach to cardiovascular wellness."},
        {"name": "Berlin Heart Specialists", "address": "Mitte, Berlin", "lat": 52.5200, "lon": 13.3900, "description": "Specializing in rhythm disorders and heart failure."},
        {"name": "Paris Cardiology Clinic", "address": "Champs-Élysées, Paris", "lat": 48.8600, "lon": 2.3300, "description": "Renowned for innovative cardiac treatments."},
    ],
    "nutritionist": [
        {"name": "Mumbai NutriBalance", "address": "101 Food Blvd, Mumbai", "lat": 19.0750, "lon": 72.8750, "description": "Personalized diet plans for heart health."},
        {"name": "New Delhi Diet & Wellness", "address": "202 Green St, New Delhi", "lat": 28.6100, "lon": 77.2050, "description": "Registered dietitians for chronic disease management."},
        {"name": "Bengaluru Diet Clinic", "address": "303 Healthy Ln, Bengaluru", "lat": 12.9700, "lon": 77.5900, "description": "Focus on sustainable and healthy eating habits."},
        {"name": "Hyderabad Nutrition Hub", "address": "Banjara Hills, Hyderabad", "lat": 17.4100, "lon": 78.4500, "description": "Weight management and therapeutic diets."},
        {"name": "Chennai Food & Health", "address": "T. Nagar, Chennai", "lat": 13.0400, "lon": 80.2300, "description": "Dietary advice for cardiovascular disease prevention."},
        {"name": "Kolkata Wellness Diet", "address": "Salt Lake City, Kolkata", "lat": 22.5800, "lon": 88.4000, "description": "Holistic nutrition for improved well-being."},
        {"name": "London Nutrition Experts", "address": "Baker Street, London", "lat": 51.5200, "lon": -0.1500, "description": "Evidence-based nutritional guidance."},
        {"name": "New York Wellness Dietitians", "address": "Central Park West, New York", "lat": 40.7800, "lon": -73.9700, "description": "Specializing in heart-healthy eating plans."},
        {"name": "Tokyo Dietitian Services", "address": "Ginza, Tokyo", "lat": 35.6700, "lon": 139.7600, "description": "Japanese and Western dietary approaches."},
        {"name": "Sydney Nutrition & Lifestyle", "address": "Bondi Beach, Sydney", "lat": -33.8900, "lon": 151.2700, "description": "Integrated nutrition and lifestyle coaching."},
        {"name": "Toronto Diet & Health", "address": "Distillery District, Toronto", "lat": 43.6500, "lon": -79.3600, "description": "Personalized meal planning and support."},
        {"name": "Berlin NutriCare", "address": "Kreuzberg, Berlin", "lat": 52.5000, "lon": 13.4000, "description": "Nutritional therapy for chronic conditions."},
        {"name": "Paris Healthy Eating", "address": "Le Marais, Paris", "lat": 48.8500, "lon": 2.3600, "description": "French culinary arts meets balanced nutrition."},
    ],
    "fitness_center": [
        {"name": "Mumbai Active Life Gym", "address": "303 Strong Rd, Mumbai", "lat": 19.0700, "lon": 72.8700, "description": "State-of-the-art gym with personal trainers."},
        {"name": "Mumbai Fitness Hub", "address": "Andheri East, Mumbai", "lat": 19.1100, "lon": 72.8600, "description": "Group classes and cardio equipment."},
        {"name": "New Delhi FitZone Studio", "address": "404 Energy Ave, New Delhi", "lat": 28.6000, "lon": 77.2000, "description": "High-intensity interval training and yoga."},
        {"name": "Bengaluru Powerhouse Gym", "address": "505 Fitness St, Bengaluru", "lat": 12.9650, "lon": 77.5850, "description": "Strength training and functional fitness."},
        {"name": "Hyderabad Cardio Fitness", "address": "Jubilee Hills, Hyderabad", "lat": 17.4300, "lon": 78.4000, "description": "Specialized cardio programs for heart health."},
        {"name": "Chennai Workout World", "address": "Adyar, Chennai", "lat": 13.0000, "lon": 80.2600, "description": "Modern gym with diverse fitness options."},
        {"name": "Kolkata Active Zone", "address": "New Town, Kolkata", "lat": 22.5600, "lon": 88.4500, "description": "Spacious gym with certified trainers."},
        {"name": "London Fitness First", "address": "Oxford Street, London", "lat": 51.5150, "lon": -0.1400, "description": "Premium fitness club with swimming pool."},
        {"name": "New York Gym & Spa", "address": "Times Square, New York", "lat": 40.7500, "lon": -73.9800, "description": "Luxury fitness and wellness center."},
        {"name": "Tokyo Health Club", "address": "Shibuya, Tokyo", "lat": 35.6600, "lon": 139.7000, "description": "Fitness programs for all ages and levels."},
        {"name": "Sydney Active Living", "address": "Darling Harbour, Sydney", "lat": -33.8700, "lon": 151.1900, "description": "Outdoor and indoor fitness activities."},
        {"name": "Toronto Fitness Hub", "address": "King Street West, Toronto", "lat": 43.6400, "lon": -79.3900, "description": "CrossFit and personalized training."},
        {"name": "Berlin Sport & Health", "address": "Prenzlauer Berg, Berlin", "lat": 52.5400, "lon": 13.4100, "description": "Diverse classes and modern equipment."},
        {"name": "Paris Gym Central", "address": "Saint-Germain-des-Prés, Paris", "lat": 48.8500, "lon": 2.3300, "description": "Boutique gym with personalized coaching."},
    ],
    "support_group": [
        {"name": "Mumbai Heart Support Group", "address": "505 Community Hall, Mumbai", "lat": 19.0800, "lon": 72.8900, "description": "Weekly meetings for heart disease patients and families."},
        {"name": "New Delhi Wellness Circle", "address": "606 Peace Centre, New Delhi", "lat": 28.6200, "lon": 77.2200, "description": "Support for managing stress and chronic conditions."},
        {"name": "Bengaluru Health & Hope", "address": "707 Unity Place, Bengaluru", "lat": 12.9800, "lon": 77.6000, "description": "Peer support for cardiovascular health."},
        {"name": "Hyderabad Patient Network", "address": "Kukatpally, Hyderabad", "lat": 17.4900, "lon": 78.4000, "description": "Connecting patients for shared experiences and advice."},
        {"name": "Chennai Recovery Group", "address": "Mylapore, Chennai", "lat": 13.0300, "lon": 80.2700, "description": "Support for post-cardiac event recovery."},
        {"name": "Kolkata Life Line", "address": "Alipore, Kolkata", "lat": 22.5100, "lon": 88.3300, "description": "Emotional and practical support for heart patients."},
        {"name": "London Health Collective", "address": "Trafalgar Square, London", "lat": 51.5000, "lon": -0.1200, "description": "Community support for various health challenges."},
        {"name": "New York Support Network", "address": "Brooklyn Bridge, New York", "lat": 40.7000, "lon": -74.0000, "description": "Groups for chronic illness management."},
        {"name": "Tokyo Patient Support", "address": "Ueno, Tokyo", "lat": 35.7100, "lon": 139.7700, "description": "Support meetings for health and well-being."},
        {"name": "Sydney Wellness Support", "address": "Manly Beach, Sydney", "lat": -33.7900, "lon": 151.2800, "description": "Mindfulness and stress reduction groups."},
        {"name": "Toronto Health Support", "address": "Queen Street West, Toronto", "lat": 43.6400, "lon": -79.4000, "description": "Peer support for healthy living."},
        {"name": "Berlin Patient Forum", "address": "Tiergarten, Berlin", "lat": 52.5100, "lon": 13.3700, "description": "Open discussions for health-related topics."},
        {"name": "Paris Bien-être Group", "address": "Montmartre, Paris", "lat": 48.8800, "lon": 2.3400, "description": "Support for holistic well-being and recovery."},
    ]
}

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on Earth using the Haversine formula.
    Returns distance in kilometers.
    """
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def find_resources_in_dataset(city_name, resource_type, max_distance_km):
    """
    Loads Indian healthcare data from a local CSV and finds nearby resources.
    """
    try:
        # Load the new dataset file
        df = pd.read_csv("india_health_facilities.csv")

        user_location = MAJOR_CITIES.get(city_name)
        if not user_location:
            st.warning("Please select a valid city from the list.")
            return pd.DataFrame()

        user_lat, user_lon = user_location['lat'], user_location['lon']

        # --- Filter for Maharashtra First ---
        # This dataset has a 'state_name' column, which is very useful.
        df_maharashtra = df[df['State Name'].str.strip().str.lower() == 'maharashtra'].copy()

        # --- Data Cleaning using the CORRECT column names from the new file ---
        required_cols = ['Latitude', 'Longitude', 'Facility Name']
        if not all(col in df_maharashtra.columns for col in required_cols):
            st.error(f"CSV file must contain the following columns: {', '.join(required_cols)}")
            return pd.DataFrame()

        df_maharashtra = df_maharashtra.dropna(subset=['Latitude', 'Longitude'])
        df_maharashtra['Latitude'] = pd.to_numeric(df_maharashtra['Latitude'], errors='coerce')
        df_maharashtra['Longitude'] = pd.to_numeric(df_maharashtra['Longitude'], errors='coerce')

        # --- Calculate Distances ---
        distances = [haversine_distance(user_lat, user_lon, row['Latitude'], row['Longitude']) for index, row in df_maharashtra.iterrows()]
        df_maharashtra['distance_km'] = distances

        # --- Filter by Distance ---
        nearby_df = df_maharashtra[df_maharashtra['distance_km'] <= max_distance_km].copy()
        nearby_df['distance_km'] = nearby_df['distance_km'].round(2)
        
        return nearby_df.sort_values('distance_km')

    except FileNotFoundError:
        st.error("Error: The 'india_health_facilities.csv' file was not found.")
        st.info("Please download the dataset from data.gov.in and place it in the project folder.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while processing the dataset: {e}")
        return pd.DataFrame()

# --- Streamlit GUI ---
def create_streamlit_gui(username):
    """
    Sets up the Streamlit application interface for the CardioHealth Risk Predictor.
    """
    # st.set_page_config(
    #     page_title="CardioHealth Risk Predictor Pro",
    #     page_icon="❤️",
    #     layout="wide",
    #     initial_sidebar_state="collapsed"
    # )
    st.sidebar.success(f"Welcome, {username}!")
    if st.sidebar.button("Logout"):
        st.session_state['authenticated'] = False
        st.session_state['username'] = ""
        st.rerun()

    # --- New and Improved CSS Styling ---
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        html, body, [class*="st-"] {
            font-family: 'Poppins', sans-serif;
            color: #E0E0E0;
        }
        
        .main {
            background-color: #1A212C;
            padding: 2.5rem;
            border-radius: 15px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
        }

        .stButton>button {
            background-color: #4B90F9;
            color: white;
            font-weight: 600;
            border-radius: 12px;
            padding: 12px 25px;
            border: none;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .stButton>button:hover {
            background-color: #3B7DD8;
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5);
        }

        .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
            border-radius: 10px;
            border: 1px solid #37475A;
            padding: 12px;
            background-color: #222E3A;
            color: #E0E0E0;
            box-shadow: inset 0 2px 5px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        
        .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus, .stSelectbox>div>div>select:focus {
            border-color: #4B90F9;
            box-shadow: 0 0 0 2px #4B90F9;
        }

        .stExpander {
            border-radius: 12px;
            border: 1px solid #37475A;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            background-color: #222E3A;
            padding: 20px;
            margin-top: 25px;
        }
        
        .stExpander > div > div > p {
            font-weight: 600;
            color: #E0E0E0;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF;
            font-weight: 600;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        }

        .stAlert {
            border-radius: 12px;
            font-size: 1.1em;
            font-weight: 600;
        }
        .stAlert.error { background-color: #C0392B; color: white; border-color: #A93226; }
        .stAlert.warning { background-color: #F39C12; color: white; border-color: #D68910; }
        .stAlert.success { background-color: #27AE60; color: white; border-color: #229954; }

        .stMarkdown p, .stMarkdown ul, .stMarkdown li {
            font-size: 16px;
            line-height: 1.7;
            color: #C0C0C0;
        }
        
        .stMarkdown li {
            font-size: 15px;
            margin-bottom: 5px;
        }

        /* Redesigning the Navigation Tabs */
        .stRadio {
            background: #222E3A;
            padding: 10px;
            border-radius: 12px;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
            margin-bottom: 20px;
        }

        .stRadio > label {
            flex-direction: row;
            justify-content: space-around;
            flex-wrap: nowrap;
            padding: 0;
            margin: 0;
        }
        
        .stRadio > label > div {
            padding: 15px 20px;
            border-radius: 10px;
            margin: 0 5px;
            font-weight: 600;
            color: #A0A0A0;
            background-color: transparent;
            transition: all 0.3s ease-in-out;
            border: 2px solid transparent;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        }
        
        .stRadio > label > div:hover {
            background-color: #2D3C4C;
            color: #FFFFFF;
        }
        
        .stRadio [data-baseweb="radio"] > div:first-child {
            display: none !important;
        }
        
        .stRadio [data-baseweb="radio"] > div:last-child {
            background-color: #4B90F9 !important;
            border-color: #4B90F9 !important;
            color: white !important;
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.6);
            text-shadow: 1px 1px 4px rgba(255,255,255,0.9);
        }

        .stGraphViz {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .media-container {
            height: 350px; /* You can adjust this height */
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            margin-bottom: 20px;
        }

        .media-container img, .media-container iframe {
            width: 100%;
            height: 100%;
            object-fit: cover; /* This makes the image fill the container without distortion */
            border: none;
        }
        
        </style>
        """,
        unsafe_allow_html=True
    )

    # st.title("❤️ CardioHealth Risk Predictor Pro")
    # st.markdown("### Advanced Cardiovascular Risk Assessment for Personalized Insights")
    # st.markdown("---")

    page_selection = st.radio(
        "Navigation",
        ["Single Prediction", "Bulk Prediction (CSV)", "Local Resources", "Risk Trend Analysis", "What-If Scenario", "Model Comparison", "Educational Hub"],
        key="top_navbar_radio",
        horizontal=True
    )
    st.markdown("---")

    if 'age' not in st.session_state:
        st.session_state.full_name = "ruturaj Nawale"
        st.session_state.phone_number = "+91 9833097874"
        st.session_state.age = 50.0
        st.session_state.height = 165.0
        st.session_state.weight = 70.0
        st.session_state.ap_hi = 120.0
        st.session_state.ap_lo = 80.0
        st.session_state.gender = "Male"
        st.session_state.cholesterol = "Normal (1)"
        st.session_state.gluc = "Normal (1)"
        st.session_state.smoke = "Non-smoker"
        st.session_state.alco = "Non-drinker"
        st.session_state.active = "Moderately Active"
        st.session_state.stress = "Moderate"
        st.session_state.prediction_made = False
        st.session_state.rec_data = {}
        st.session_state.bulk_prediction_made = False
        st.session_state.bulk_results_df = None
        st.session_state.resource_location = "Mumbai"
        st.session_state.resource_type = "cardiologist"
        st.session_state.prediction_history = []
        st.session_state.user_input_for_report = {}
        st.session_state.what_if_input = {
            'age': 50.0, 'height': 165.0, 'weight': 70.0,
            'ap_hi': 120.0, 'ap_lo': 80.0, 'gender': "Male",
            'cholesterol': "Normal (1)", 'gluc': "Normal (1)",
            'smoke': "Non-smoker", 'alco': "Non-drinker",
            'active': "Moderately Active", 'stress': "Moderate"
        }
        st.session_state.what_if_prob = None
        st.session_state.what_if_result = None
        st.session_state.what_if_baseline_prob = None
        st.session_state.what_if_results_to_show = False
        st.session_state.compare_results = None


    if page_selection == "Single Prediction":
        st.header("Patient Information (Single Prediction)")
        st.markdown("Please enter the patient's details below:")
        st.markdown("---")
        # <<< MODIFICATION: Add new input fields for patient details
        st.subheader("Personal Details")
        with st.container(border=True):
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.session_state.full_name = st.text_input(
                    "Patient Full Name",
                    value=st.session_state.full_name,
                    key="input_full_name",
                    help="Enter the patient's full legal name."
                )
            with p_col2:
                st.session_state.phone_number = st.text_input(
                    "Patient Phone Number / ID",
                    value=st.session_state.phone_number,
                    key="input_phone",
                    help="Enter a unique identifier like a phone number or patient ID."
                )
        st.subheader("Health Metrics")
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.session_state.age = st.number_input(
                    "Age (years)",
                    value=st.session_state.age,
                    min_value=1.0, max_value=120.0, step=1.0, format="%.0f",
                    help="Enter age in full years.",
                    key="input_age"
                )
                if st.session_state.age < 18:
                    st.warning("⚠️ This app is primarily for adult cardiovascular risk assessment. Please consult a pediatrician for minors.")
                
                st.session_state.height = st.number_input(
                    "Height (cm)",
                    value=st.session_state.height,
                    min_value=50.0, max_value=250.0, step=1.0, format="%.0f",
                    help="Enter height in centimeters.",
                    key="input_height"
                )
                st.session_state.weight = st.number_input(
                    "Weight (kg)",
                    value=st.session_state.weight,
                    min_value=10.0, max_value=300.0, step=0.1, format="%.1f",
                    help="Enter weight in kilograms.",
                    key="input_weight"
                )
                
            with col2:
                st.session_state.ap_hi = st.number_input(
                    "Systolic BP (mmHg)",
                    value=st.session_state.ap_hi,
                    min_value=50.0, max_value=300.0, step=1.0, format="%.0f",
                    help="Enter the top number of your blood pressure reading.",
                    key="input_ap_hi"
                )
                if st.session_state.ap_hi > 180:
                    st.error("🚨 Critically High Systolic BP. Seek immediate medical attention.")
                elif st.session_state.ap_hi > 130:
                    st.warning("⚠️ Systolic BP is in the elevated range (Hypertension Stage 1 or 2).")

                st.session_state.ap_lo = st.number_input(
                    "Diastolic BP (mmHg)",
                    value=st.session_state.ap_lo,
                    min_value=30.0, max_value=200.0, step=1.0, format="%.0f",
                    help="Enter the bottom number of your blood pressure reading.",
                    key="input_ap_lo"
                )
                if st.session_state.ap_lo > 120:
                    st.error("🚨 Critically High Diastolic BP. Seek immediate medical attention.")
                elif st.session_state.ap_lo > 85:
                    st.warning("⚠️ Diastolic BP is in the elevated range.")

                st.session_state.gender = st.selectbox(
                    "Gender",
                    ["Male", "Female"],
                    index=["Male", "Female"].index(st.session_state.gender),
                    help="Select biological gender.",
                    key="input_gender"
                )

            with col3:
                st.session_state.cholesterol = st.selectbox(
                    "Cholesterol Level",
                    ["Normal (1)", "Above Normal (2)", "Well Above (3)"],
                    index=["Normal (1)", "Above Normal (2)", "Well Above (3)"].index(st.session_state.cholesterol),
                    help="1: Normal, 2: Above Normal, 3: Well Above Normal.",
                    key="input_cholesterol"
                )
                st.session_state.gluc = st.selectbox(
                    "Glucose Level",
                    ["Normal (1)", "Above Normal (2)", "Well Above (3)"],
                    index=["Normal (1)", "Above Normal (2)", "Well Above (3)"].index(st.session_state.gluc),
                    help="1: Normal, 2: Above Normal, 3: Well Above Normal.",
                    key="input_gluc"
                )
                st.session_state.smoke = st.selectbox(
                    "Smoking Status",
                    ["Non-smoker", "Smoker"],
                    index=["Non-smoker", "Smoker"].index(st.session_state.smoke),
                    help="Are you currently a smoker?",
                    key="input_smoke"
                )

            col4, col5, col6 = st.columns(3)
            with col4:
                st.session_state.alco = st.selectbox(
                    "Alcohol Intake",
                    ["Non-drinker", "Moderate Drinker", "Heavy Drinker"],
                    index=["Non-drinker", "Moderate Drinker", "Heavy Drinker"].index(st.session_state.alco),
                    help="Select your typical alcohol consumption level.",
                    key="input_alco"
                )
            with col5:
                st.session_state.active = st.selectbox(
                    "Physical Activity",
                    ["Sedentary", "Moderately Active", "Very Active"],
                    index=["Sedentary", "Moderately Active", "Very Active"].index(st.session_state.active),
                    help="Sedentary: little to no exercise; Moderately Active: some regular exercise; Very Active: intense regular exercise.",
                    key="input_active"
                )
            with col6:
                st.session_state.stress = st.selectbox(
                    "Stress Level",
                    ["Low", "Moderate", "High"],
                    index=["Low", "Moderate", "High"].index(st.session_state.stress),
                    help="Your perceived stress level (e.g., Low: rarely stressed, High: frequently stressed).",
                    key="input_stress"
                )

        st.markdown("---")
        btn_col1, btn_col2 = st.columns([1, 1])

        with btn_col1:
            if st.button("Analyze Cardiovascular Risk", use_container_width=True):
                # --- Validation logic before prediction ---
                valid_inputs = True
                
                # Blood Pressure Validation
                if st.session_state.ap_lo >= st.session_state.ap_hi:
                    st.error("Error: Diastolic BP (ap_lo) must be less than Systolic BP (ap_hi).")
                    valid_inputs = False
                    
                if st.session_state.ap_lo < 30 or st.session_state.ap_lo > 200:
                    st.error("Error: Diastolic BP is out of expected range (30-200).")
                    valid_inputs = False
                
                if st.session_state.ap_hi < 50 or st.session_state.ap_hi > 300:
                    st.error("Error: Systolic BP is out of expected range (50-300).")
                    valid_inputs = False
                
                try:
                    bmi = st.session_state.weight / (st.session_state.height / 100)**2
                    if bmi < 10 or bmi > 100: # Simple BMI range check
                        st.error("Error: Calculated BMI is outside a realistic range. Please check height and weight.")
                        valid_inputs = False
                except ZeroDivisionError:
                    st.error("Error: Height cannot be zero.")
                    valid_inputs = False

                if valid_inputs:
                    try:
                        cholesterol_map = {"Normal (1)": 1, "Above Normal (2)": 2, "Well Above (3)": 3}
                        gluc_map = {"Normal (1)": 1, "Above Normal (2)": 2, "Well Above (3)": 3}
                        smoke_map = {"Non-smoker": 0, "Smoker": 1}
                        alco_map = {"Non-drinker": 0, "Moderate Drinker": 1, "Heavy Drinker": 2}
                        active_map = {"Sedentary": 0, "Moderately Active": 1, "Very Active": 2}
                        stress_map = {"Low": 0, "Moderate": 1, "High": 2}

                        user_input = {
                            'full_name': st.session_state.full_name.strip(),
                            'phone_number': st.session_state.phone_number.strip(),
                            'age': st.session_state.age,
                            'height': st.session_state.height,
                            'weight': st.session_state.weight,
                            'ap_hi': st.session_state.ap_hi,
                            'ap_lo': st.session_state.ap_lo,
                            'gender': st.session_state.gender,
                            'cholesterol': cholesterol_map[st.session_state.cholesterol],
                            'gluc': gluc_map[st.session_state.gluc],
                            'smoke': smoke_map[st.session_state.smoke],
                            'alco': alco_map[st.session_state.alco],
                            'active': active_map[st.session_state.active],
                            'stress': stress_map[st.session_state.stress]
                        }
                        st.session_state.user_input_for_report = user_input

                        result, prob = predict_disease(user_input)
                        st.session_state.rec_data = generate_recommendations(user_input, prob, result)
                        st.session_state.prediction_made = True
                        st.session_state.bulk_prediction_made = False
                        st.session_state.bulk_results_df = None
                        db.add_prediction(
                            username=username,
                            patient_name=st.session_state.full_name.strip(),
                            patient_phone=st.session_state.phone_number.strip(),
                            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                            probability=prob,
                            risk_level=st.session_state.rec_data['risk_level'],
                            source="Single Prediction"
                        )
                        st.success("Analysis Complete and Patient Record Saved!")
                    except Exception as e:
                        st.error(f"An error occurred during analysis: {e}")

                #         st.session_state.prediction_history.append({
                #             "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                #             "probability": prob,
                #             "risk_level": st.session_state.rec_data['risk_level'],
                #             "source": "Single Prediction"
                #         })
                #         st.success("Analysis Complete! See results below.")
                #     except Exception as e:
                #         st.error(f"An error occurred during analysis: {e}. Please ensure all inputs are valid.")
                #         st.session_state.prediction_made = False
                # else:
                #     st.error("Please correct the input errors and try again.")
                
        with btn_col2:
            if st.button("Reset Form", use_container_width=True):
                st.session_state.full_name = "Ruturaj Nawale"
                st.session_state.phone_number = "+91 98330 97874"
                st.session_state.age = 50.0
                st.session_state.height = 165.0
                st.session_state.weight = 70.0
                st.session_state.ap_hi = 120.0
                st.session_state.ap_lo = 80.0
                st.session_state.gender = "Male"
                st.session_state.cholesterol = "Normal (1)"
                st.session_state.gluc = "Normal (1)"
                st.session_state.smoke = "Non-smoker"
                st.session_state.alco = "Non-drinker"
                st.session_state.active = "Moderately Active"
                st.session_state.stress = "Moderate"
                st.session_state.prediction_made = False
                st.session_state.rec_data = {}
                st.session_state.bulk_prediction_made = False
                st.session_state.bulk_results_df = None
                st.session_state.prediction_history = []
                st.rerun()

        st.markdown("---")

        if st.session_state.prediction_made:
            st.header("Advanced Risk Analysis & Recommendations")
            risk_level_text = st.session_state.rec_data['risk_level']
            risk_color = st.session_state.rec_data['risk_color']

            if risk_color == "red":
                st.error(f"**Risk Assessment: {risk_level_text}**")
            elif risk_color == "orange":
                st.warning(f"**Risk Assessment: {risk_level_text}**")
            else:
                st.success(f"**Risk Assessment: {risk_level_text}**")

            st.markdown(f"**{st.session_state.rec_data['probability']}**")
            st.markdown("---")
            pdf_data = generate_pdf_report(st.session_state.user_input_for_report, st.session_state.rec_data)
            st.download_button(
                label="Download Full Report (PDF)",
                data=pdf_data,
                file_name=f"CardioHealth_Report_{datetime.date.today()}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

            st.subheader("Cardiovascular Disease Risk Proportion")
            
            prob_value = st.session_state.rec_data['prob_value']
            
            # Create a DataFrame for Plotly
            df_pie = pd.DataFrame({
                'Category': ['No Disease Risk', 'Cardiovascular Disease Risk'],
                'Value': [1 - prob_value, prob_value]
            })

            # Create an interactive donut chart
            fig_donut = px.pie(
                df_pie, 
                names='Category', 
                values='Value',
                title='Predicted Risk Distribution',
                hole=0.4, # This creates the donut shape
                color_discrete_map={
                    'No Disease Risk': '#27AE60',
                    'Cardiovascular Disease Risk': '#C0392B'
                }
            )

            # Update layout for a cleaner look that matches your theme
            fig_donut.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#E0E0E0',
                title_font_size=20,
                legend_title_text='',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                )
            )
            
            # Update traces for better hover info and text display
            fig_donut.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                hoverinfo='label+percent',
                marker=dict(line=dict(color='#1A212C', width=4))
            )

            st.plotly_chart(fig_donut, use_container_width=True)
            st.markdown("---")
            
            with st.expander("Key Risk Factors Identified", expanded=True):
                st.markdown(st.session_state.rec_data['factors'])

            with st.expander("Personalized Recommendations"):
                st.markdown(st.session_state.rec_data['recommendations'])

            with st.expander("Screening & Monitoring Plan"):
                st.markdown(st.session_state.rec_data['screening'])

            with st.expander("Educational Resources"):
                st.markdown(st.session_state.rec_data['resources'])

    # --- Bulk Prediction Page ---
    elif page_selection == "Bulk Prediction (CSV)":
        st.header("Bulk Cardiovascular Risk Prediction")
        st.markdown("Upload a CSV file containing patient data for bulk analysis.")
        st.markdown("""
        **CSV Requirements:**
        Your CSV file must contain the following columns with exact names and expected values:
        `age` (years), `gender` (`Male`/`Female` or `0`/`1`), `height` (cm), `weight` (kg),
        `ap_hi` (Systolic BP), `ap_lo` (Diastolic BP),
        `cholesterol` (`Normal (1)`/`Above Normal (2)`/`Well Above (3)` or `1`/`2`/`3`),
        `gluc` (`Normal (1)`/`Above Normal (2)`/`Well Above (3)` or `1`/`2`/`3`),
        `smoke` (`Non-smoker`/`Smoker` or `0`/`1`),
        `alco` (`Non-drinker`/`Moderate Drinker`/`Heavy Drinker` or `0`/`1`/`2`),
        `active` (`Sedentary`/`Moderately Active`/`Very Active` or `0`/`1`/`2`).

        An optional `id` or `stress` column can be present but will be ignored for prediction.
        """)
        st.markdown("---")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            try:
                df_uploaded = pd.read_csv(uploaded_file)
                st.write("Uploaded Data Preview:")
                st.dataframe(df_uploaded.head())

                # --- NEW FEATURE: EDA Visualizations ---
                with st.expander("Explore Your Data with Interactive Charts", expanded=False):
                    st.subheader("Data Insights")
                    plots = generate_bulk_eda_plots(df_uploaded)

                    if plots:
                        for plot_fig in plots:
                            st.plotly_chart(plot_fig, use_container_width=True)
                    else:
                        st.info("No plots could be generated from the uploaded data. Please check your column names.")
                # --- END OF NEW FEATURE ---

                if st.button("Run Bulk Prediction", use_container_width=True):
                    with st.spinner("Processing CSV for predictions..."):
                        st.session_state.prediction_made = False
                        st.session_state.rec_data = {}

                        predicted_df = bulk_predict_disease(df_uploaded.copy())

                        if predicted_df is not None:
                            st.session_state.bulk_results_df = predicted_df
                            st.session_state.bulk_prediction_made = True
                            st.success("Bulk prediction complete! See results below.")

                            current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                            avg_prob = predicted_df['Prediction_Probability'].mean()
                            avg_risk_level = "High Risk" if avg_prob > 0.75 else ("Moderate Risk" if avg_prob > 0.45 else "Low Risk")

                            st.session_state.prediction_history.append({
                                "timestamp": current_timestamp,
                                "probability": avg_prob,
                                "risk_level": avg_risk_level,
                                "source": f"Bulk Avg ({len(predicted_df)} records)"
                            })
                        else:
                            st.session_state.bulk_prediction_made = False

            except pd.errors.EmptyDataError:
                st.error("The uploaded CSV file is empty. Please upload a file with data.")
                st.session_state.bulk_prediction_made = False
            except pd.errors.ParserError:
                st.error("Could not parse the CSV file. Please check if it's a valid CSV format.")
                st.session_state.bulk_prediction_made = False
            except Exception as e:
                st.error(f"An unexpected error occurred while reading or processing the CSV: {e}")
                st.session_state.bulk_prediction_made = False

        if st.session_state.bulk_prediction_made and st.session_state.bulk_results_df is not None:
            st.subheader("Bulk Prediction Results")
            st.dataframe(st.session_state.bulk_results_df)

            csv_output = st.session_state.bulk_results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv_output,
                file_name="cardio_bulk_predictions.csv",
                mime="text/csv",
                help="Download the table above as a CSV file."
            )
            st.markdown("---")

    # --- Local Resources Page ---
    elif page_selection == "Local Resources":
       st.header("Find Nearby Health Resources")
       st.markdown("Discover healthcare providers in **Maharashtra, India** from a public dataset.")
       st.markdown("---")

       selected_city = st.selectbox(
           "Select Your City in Maharashtra",
           options=list(MAJOR_CITIES.keys())
        )

       resource_type = st.selectbox(
           "Select Resource Type",
           ["Hospital", "Clinic", "Health Centre"],
           index=0
       )

       max_distance = st.slider(
           "Search Radius (km)",
           min_value=1, max_value=50, value=10, step=1
       )

       if st.button("Search in Dataset", use_container_width=True):
           with st.spinner(f"Searching for resources near '{selected_city}'..."):
               nearby_df = find_resources_in_dataset(selected_city, resource_type, max_distance)

               if not nearby_df.empty:
                   st.subheader(f"Found {len(nearby_df)} resources:")
                   
                   # Update column names here to match the new CSV for a cleaner display
                   display_cols = ['Facility Name', 'District Name', 'Facility Type', 'Facility Address', 'distance_km']
                   st.dataframe(nearby_df[display_cols])
   
                   st.subheader("Locations on Map")
                   # Use the correct column names for the map as well
                   map_data = nearby_df[['Latitude', 'Longitude']].dropna().rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})
                   st.map(map_data, zoom=9)
               else:
                   st.info(f"No resources found within {max_distance} km of {selected_city}.")

    # --- Risk Trend Analysis Page (Modified for Personalized Trajectory) ---
    elif page_selection == "Risk Trend Analysis":
        st.header("Personalized Risk Trajectory Over Time")
        st.markdown("Track your cardiovascular risk by adding historical data points and viewing the trend.")
        st.markdown("---")
        
        # --- Add new data point form ---
        with st.expander("Add a Past Data Point"):
            add_col1, add_col2, add_col3 = st.columns(3)
            with add_col1:
                past_age = st.number_input("Age (years)", min_value=1.0, max_value=120.0, step=1.0, format="%.0f", key="past_age")
                past_height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, step=1.0, format="%.0f", key="past_height")
                past_weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, step=0.1, format="%.1f", key="past_weight")
            with add_col2:
                past_ap_hi = st.number_input("Systolic BP (mmHg)", min_value=50.0, max_value=300.0, step=1.0, format="%.0f", key="past_ap_hi")
                past_ap_lo = st.number_input("Diastolic BP (mmHg)", min_value=30.0, max_value=200.0, step=1.0, format="%.0f", key="past_ap_lo")
                past_gender = st.selectbox("Gender", ["Male", "Female"], key="past_gender")
            with add_col3:
                past_date = st.date_input("Date of Measurement", value=datetime.date.today(), key="past_date")
                past_cholesterol = st.selectbox("Cholesterol Level", ["Normal (1)", "Above Normal (2)", "Well Above (3)"], key="past_cholesterol")
                
            if st.button("Add to History", use_container_width=True):
                try:
                    cholesterol_map = {"Normal (1)": 1, "Above Normal (2)": 2, "Well Above (3)": 3}
                    gluc_map = {"Normal (1)": 1, "Above Normal (2)": 2, "Well Above (3)": 3}
                    smoke_map = {"Non-smoker": 0, "Smoker": 1}
                    alco_map = {"Non-drinker": 0, "Moderate Drinker": 1, "Heavy Drinker": 2}
                    active_map = {"Sedentary": 0, "Moderately Active": 1, "Very Active": 2}
                    stress_map = {"Low": 0, "Moderate": 1, "High": 2}

                    past_input = {
                        'age': float(past_age), 'height': float(past_height), 'weight': float(past_weight),
                        'ap_hi': float(past_ap_hi), 'ap_lo': float(past_ap_lo), 'gender': past_gender,
                        'cholesterol': cholesterol_map[past_cholesterol], 'gluc': 1, # Defaulting gluc/other values for simplicity
                        'smoke': 0, 'alco': 0, 'active': 1, 'stress': 1
                    }
                    _, prob = predict_disease(past_input)
                    risk_level = "High Risk" if prob > 0.75 else ("Moderate Risk" if prob > 0.45 else "Low Risk")
                    
                    st.session_state.prediction_history.append({
                        "timestamp": past_date,
                        "probability": prob,
                        "risk_level": risk_level,
                        "source": "Manual Entry"
                    })
                    st.success(f"Added new data point for {past_date} with a risk of {prob*100:.1f}%.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding data: {e}. Please check your inputs.")
        
        st.markdown("---")

        if st.session_state.prediction_history:
            history_df = pd.DataFrame(st.session_state.prediction_history)
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            history_df = history_df.sort_values(by='timestamp')

            # Use Plotly for an interactive line chart
            fig = px.line(
                history_df,
                x='timestamp',
                y='probability',
                color='source',
                title='Personalized Risk Trend Over Time',
                labels={'timestamp': 'Date of Measurement', 'probability': 'Risk Probability'},
                markers=True,
                color_discrete_map={"Single Prediction": "#4B90F9", "Manual Entry": "#E5B23F", "Bulk Avg": "#70D6A7"}
            )
            fig.add_hrect(y0=0.75, y1=1, line_width=0, fillcolor="#C0392B", opacity=0.1, annotation_text="High Risk Zone")
            fig.add_hrect(y0=0.45, y1=0.75, line_width=0, fillcolor="#F39C12", opacity=0.1, annotation_text="Moderate Risk Zone")
            fig.add_hrect(y0=0, y1=0.45, line_width=0, fillcolor="#27AE60", opacity=0.1, annotation_text="Low Risk Zone")

            fig.update_layout(
                plot_bgcolor='#222E3A',
                paper_bgcolor='#222E3A',
                font_color='#E0E0E0',
                title_font_color='#E0E0E0',
                yaxis_range=[0, 1]
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("Prediction History Details")
            st.dataframe(history_df[['timestamp', 'risk_level', 'probability', 'source']].style.format({"probability": "{:.1%}"}))
            st.info("Note: This history is stored only for your current session and will reset if the app is reloaded.")

        else:
            st.info("No predictions have been made yet to display a trend. Please use the 'Single Prediction' page or add a past data point above.")
        st.markdown("---")

    # --- What-If Scenario Page ---
    elif page_selection == "What-If Scenario":
        st.header("Interactive 'What-If' Scenario Analysis")
        st.markdown("Adjust the parameters below to see how hypothetical changes might impact the cardiovascular risk prediction.")
        st.markdown("---")

        st.subheader("Adjust Health Parameters")
        with st.container(border=True):
            col_wi1, col_wi2, col_wi3 = st.columns(3)

            with col_wi1:
                st.session_state.what_if_input['age'] = st.number_input(
                    "Age (years)",
                    value=st.session_state.what_if_input['age'],
                    min_value=1.0, max_value=120.0, step=1.0, format="%.0f", key="wi_age"
                )
                st.session_state.what_if_input['height'] = st.number_input(
                    "Height (cm)",
                    value=st.session_state.what_if_input['height'],
                    min_value=50.0, max_value=250.0, step=1.0, format="%.0f", key="wi_height"
                )
                st.session_state.what_if_input['weight'] = st.number_input(
                    "Weight (kg)",
                    value=st.session_state.what_if_input['weight'],
                    min_value=10.0, max_value=300.0, step=0.1, format="%.1f", key="wi_weight"
                )

            with col_wi2:
                st.session_state.what_if_input['ap_hi'] = st.number_input(
                    "Systolic BP (mmHg)",
                    value=st.session_state.what_if_input['ap_hi'],
                    min_value=50.0, max_value=300.0, step=1.0, format="%.0f", key="wi_ap_hi"
                )
                st.session_state.what_if_input['ap_lo'] = st.number_input(
                    "Diastolic BP (mmHg)",
                    value=st.session_state.what_if_input['ap_lo'],
                    min_value=30.0, max_value=200.0, step=1.0, format="%.0f", key="wi_ap_lo"
                )
                st.session_state.what_if_input['gender'] = st.selectbox(
                    "Gender",
                    ["Male", "Female"],
                    index=["Male", "Female"].index(st.session_state.what_if_input['gender']), key="wi_gender"
                )

            with col_wi3:
                st.session_state.what_if_input['cholesterol'] = st.selectbox(
                    "Cholesterol Level",
                    ["Normal (1)", "Above Normal (2)", "Well Above (3)"],
                    index=["Normal (1)", "Above Normal (2)", "Well Above (3)"].index(st.session_state.what_if_input['cholesterol']), key="wi_cholesterol"
                )
                st.session_state.what_if_input['gluc'] = st.selectbox(
                    "Glucose Level",
                    ["Normal (1)", "Above Normal (2)", "Well Above (3)"],
                    index=["Normal (1)", "Above Normal (2)", "Well Above (3)"].index(st.session_state.what_if_input['gluc']), key="wi_gluc"
                )
                st.session_state.what_if_input['smoke'] = st.selectbox(
                    "Smoking Status",
                    ["Non-smoker", "Smoker"],
                    index=["Non-smoker", "Smoker"].index(st.session_state.what_if_input['smoke']), key="wi_smoke"
                )

            col_wi4, col_wi5, col_wi6 = st.columns(3)
            with col_wi4:
                st.session_state.what_if_input['alco'] = st.selectbox(
                    "Alcohol Intake",
                    ["Non-drinker", "Moderate Drinker", "Heavy Drinker"],
                    index=["Non-drinker", "Moderate Drinker", "Heavy Drinker"].index(st.session_state.what_if_input['alco']), key="wi_alco"
                )
            with col_wi5:
                st.session_state.what_if_input['active'] = st.selectbox(
                    "Physical Activity",
                    ["Sedentary", "Moderately Active", "Very Active"],
                    index=["Sedentary", "Moderately Active", "Very Active"].index(st.session_state.what_if_input['active']), key="wi_active"
                )
            with col_wi6:
                st.session_state.what_if_input['stress'] = st.selectbox(
                    "Stress Level",
                    ["Low", "Moderate", "High"],
                    index=["Low", "Moderate", "High"].index(st.session_state.what_if_input['stress']), key="wi_stress"
                )

        st.markdown("---")

        col_wibtn1, col_wibtn2 = st.columns([1,1])
        with col_wibtn1:
            if st.button("Set as Baseline", use_container_width=True, key="run_baseline"):
                try:
                    cholesterol_map = {"Normal (1)": 1, "Above Normal (2)": 2, "Well Above (3)": 3}
                    gluc_map = {"Normal (1)": 1, "Above Normal (2)": 2, "Well Above (3)": 3}
                    smoke_map = {"Non-smoker": 0, "Smoker": 1}
                    alco_map = {"Non-drinker": 0, "Moderate Drinker": 1, "Heavy Drinker": 2}
                    active_map = {"Sedentary": 0, "Moderately Active": 1, "Very Active": 2}
                    stress_map = {"Low": 0, "Moderate": 1, "High": 2}
    
                    current_what_if_input = {
                        'age': st.session_state.what_if_input['age'],
                        'height': st.session_state.what_if_input['height'],
                        'weight': st.session_state.what_if_input['weight'],
                        'ap_hi': st.session_state.what_if_input['ap_hi'],
                        'ap_lo': st.session_state.what_if_input['ap_lo'],
                        'gender': st.session_state.what_if_input['gender'],
                        'cholesterol': cholesterol_map[st.session_state.what_if_input['cholesterol']],
                        'gluc': gluc_map[st.session_state.what_if_input['gluc']],
                        'smoke': smoke_map[st.session_state.what_if_input['smoke']],
                        'alco': alco_map[st.session_state.what_if_input['alco']],
                        'active': active_map[st.session_state.what_if_input['active']],
                        'stress': stress_map[st.session_state.what_if_input['stress']]
                    }
                    _, baseline_prob = predict_disease(current_what_if_input)
                    st.session_state.what_if_baseline_prob = baseline_prob
                    st.session_state.what_if_results_to_show = True
                    st.success(f"Baseline set! Initial risk is {baseline_prob*100:.1f}%. Now adjust inputs to see the impact.")
                except Exception as e:
                    st.error(f"An error occurred while setting baseline: {e}. Please ensure all inputs are valid.")
                    st.session_state.what_if_baseline_prob = None
                    st.session_state.what_if_results_to_show = False
        
        with col_wibtn2:
            if st.button("Reset What-If", use_container_width=True, key="reset_what_if"):
                st.session_state.what_if_input = {
                    'age': 50.0, 'height': 165.0, 'weight': 70.0,
                    'ap_hi': 120.0, 'ap_lo': 80.0, 'gender': "Male",
                    'cholesterol': "Normal (1)", 'gluc': "Normal (1)",
                    'smoke': "Non-smoker", 'alco': "Non-drinker",
                    'active': "Moderately Active", 'stress': "Moderate"
                }
                st.session_state.what_if_baseline_prob = None
                st.session_state.what_if_results_to_show = False
                st.rerun()

        if st.session_state.what_if_results_to_show and st.session_state.what_if_baseline_prob is not None:
            try:
                cholesterol_map = {"Normal (1)": 1, "Above Normal (2)": 2, "Well Above (3)": 3}
                gluc_map = {"Normal (1)": 1, "Above Normal (2)": 2, "Well Above (3)": 3}
                smoke_map = {"Non-smoker": 0, "Smoker": 1}
                alco_map = {"Non-drinker": 0, "Moderate Drinker": 1, "Heavy Drinker": 2}
                active_map = {"Sedentary": 0, "Moderately Active": 1, "Very Active": 2}
                stress_map = {"Low": 0, "Moderate": 1, "High": 2}

                current_what_if_input = {
                    'age': st.session_state.what_if_input['age'],
                    'height': st.session_state.what_if_input['height'],
                    'weight': st.session_state.what_if_input['weight'],
                    'ap_hi': st.session_state.what_if_input['ap_hi'],
                    'ap_lo': st.session_state.what_if_input['ap_lo'],
                    'gender': st.session_state.what_if_input['gender'],
                    'cholesterol': cholesterol_map[st.session_state.what_if_input['cholesterol']],
                    'gluc': gluc_map[st.session_state.what_if_input['gluc']],
                    'smoke': smoke_map[st.session_state.what_if_input['smoke']],
                    'alco': alco_map[st.session_state.what_if_input['alco']],
                    'active': active_map[st.session_state.what_if_input['active']],
                    'stress': stress_map[st.session_state.what_if_input['stress']]
                }
                
                what_if_result, what_if_prob = predict_disease(current_what_if_input)
                
                st.markdown("---")
                st.subheader("Dynamic Scenario Analysis")
                
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    st.metric("Baseline Risk", f"{st.session_state.what_if_baseline_prob*100:.1f}%")
                with col_res2:
                    st.metric("Hypothetical Risk", f"{what_if_prob*100:.1f}%")
                with col_res3:
                    risk_change_abs = (what_if_prob - st.session_state.what_if_baseline_prob) * 100
                    st.metric("Risk Change", f"{risk_change_abs:.1f}%", delta=f"{risk_change_abs:.1f}%")
                
                st.subheader("Visual Comparison")
                comparison_df = pd.DataFrame({
                    'Scenario': ['Baseline', 'Hypothetical'],
                    'Risk Probability': [st.session_state.what_if_baseline_prob, what_if_prob]
                })
                fig_compare = px.bar(
                    comparison_df,
                    x='Scenario',
                    y='Risk Probability',
                    title='Risk Probability Comparison',
                    color='Scenario',
                    color_discrete_map={'Baseline': '#4B90F9', 'Hypothetical': '#E5B23F'}
                )
                fig_compare.update_layout(
                    plot_bgcolor='#222E3A',
                    paper_bgcolor='#222E3A',
                    font_color='#E0E0E0',
                    title_font_color='#E0E0E0',
                    yaxis_range=[0, 1]
                )
                st.plotly_chart(fig_compare, use_container_width=True)
                
            except Exception as e:
                st.error(f"An error occurred during dynamic analysis: {e}")
        else:
            st.info("Set a baseline prediction first by entering your data and clicking 'Set as Baseline'.")
        
        st.markdown("---")
        
    # --- New Model Comparison Page ---
    elif page_selection == "Model Comparison":
        st.header("Compare Model Performance")
        st.markdown("Compare the predictions of the primary model with a mock comparison model.")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model Information")
            st.markdown("Here, you can see the performance characteristics of the models. These are static for this demo.")
            st.table(pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall"],
                "Primary Model": ["91.2%", "89.5%", "92.1%"],
                "Comparison Model": ["85.4%", "81.2%", "88.0%"]
            }))

        with col2:
            st.subheader("Patient Input for Comparison")
            with st.container(border=True):
                # Using unique keys for each widget on this page
                user_input_compare = {
                    'age': st.number_input("Age (years)", value=st.session_state.age, min_value=1.0, max_value=120.0, step=1.0, format="%.0f", key="comp_age"),
                    'height': st.number_input("Height (cm)", value=st.session_state.height, min_value=50.0, max_value=250.0, step=1.0, format="%.0f", key="comp_height"),
                    'weight': st.number_input("Weight (kg)", value=st.session_state.weight, min_value=10.0, max_value=300.0, step=0.1, format="%.1f", key="comp_weight"),
                    'ap_hi': st.number_input("Systolic BP (mmHg)", value=st.session_state.ap_hi, min_value=50.0, max_value=300.0, step=1.0, format="%.0f", key="comp_ap_hi"),
                    'ap_lo': st.number_input("Diastolic BP (mmHg)", value=st.session_state.ap_lo, min_value=30.0, max_value=200.0, step=1.0, format="%.0f", key="comp_ap_lo"),
                    'gender': st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(st.session_state.gender), key="comp_gender"),
                    'cholesterol': st.selectbox("Cholesterol Level", ["Normal (1)", "Above Normal (2)", "Well Above (3)"], index=["Normal (1)", "Above Normal (2)", "Well Above (3)"].index(st.session_state.cholesterol), key="comp_chol"),
                    'gluc': st.selectbox("Glucose Level", ["Normal (1)", "Above Normal (2)", "Well Above (3)"], index=["Normal (1)", "Above Normal (2)", "Well Above (3)"].index(st.session_state.gluc), key="comp_gluc"),
                    'smoke': st.selectbox("Smoking Status", ["Non-smoker", "Smoker"], index=["Non-smoker", "Smoker"].index(st.session_state.smoke), key="comp_smoke"),
                    'alco': st.selectbox("Alcohol Intake", ["Non-drinker", "Moderate Drinker", "Heavy Drinker"], index=["Non-drinker", "Moderate Drinker", "Heavy Drinker"].index(st.session_state.alco), key="comp_alco"),
                    'active': st.selectbox("Physical Activity", ["Sedentary", "Moderately Active", "Very Active"], index=["Sedentary", "Moderately Active", "Very Active"].index(st.session_state.active), key="comp_active"),
                    'stress': st.selectbox("Stress Level", ["Low", "Moderate", "High"], index=["Low", "Moderate", "High"].index(st.session_state.stress), key="comp_stress")
                }
        
        st.markdown("---")
        
        if st.button("Compare Predictions", use_container_width=True):
            try:
                cholesterol_map = {"Normal (1)": 1, "Above Normal (2)": 2, "Well Above (3)": 3}
                gluc_map = {"Normal (1)": 1, "Above Normal (2)": 2, "Well Above (3)": 3}
                smoke_map = {"Non-smoker": 0, "Smoker": 1}
                alco_map = {"Non-drinker": 0, "Moderate Drinker": 1, "Heavy Drinker": 2}
                active_map = {"Sedentary": 0, "Moderately Active": 1, "Very Active": 2}
                
                preprocessed_input = {
                    'age': user_input_compare['age'],
                    'height': user_input_compare['height'],
                    'weight': user_input_compare['weight'],
                    'ap_hi': user_input_compare['ap_hi'],
                    'ap_lo': user_input_compare['ap_lo'],
                    'gender': user_input_compare['gender'],
                    'cholesterol': cholesterol_map[user_input_compare['cholesterol']],
                    'gluc': gluc_map[user_input_compare['gluc']],
                    'smoke': smoke_map[user_input_compare['smoke']],
                    'alco': alco_map[user_input_compare['alco']],
                    'active': active_map[user_input_compare['active']],
                    'stress': user_input_compare['stress']
                }

                # Get predictions from both models
                result_primary, prob_primary = predict_disease(preprocessed_input)
                result_mock, prob_mock = mock_predict_disease(preprocessed_input)

                # Store results for display
                st.session_state.compare_results = {
                    'primary': {'result': result_primary, 'prob': prob_primary},
                    'mock': {'result': result_mock, 'prob': prob_mock},
                }
            except Exception as e:
                st.error(f"An error occurred during comparison: {e}. Please ensure all inputs are valid.")
                st.session_state.compare_results = None
        
        if 'compare_results' in st.session_state and st.session_state.compare_results:
            st.subheader("Comparison Results")
            comp_res = st.session_state.compare_results
            
            # Display predictions and probabilities
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.info("**Primary Model**")
                st.metric(label="Predicted Risk", value=f"{comp_res['primary']['prob']*100:.1f}%")
                st.markdown(f"**Prediction:** {comp_res['primary']['result']}")
            with res_col2:
                st.info("**Comparison Model**")
                st.metric(label="Predicted Risk", value=f"{comp_res['mock']['prob']*100:.1f}%")
                st.markdown(f"**Prediction:** {comp_res['mock']['result']}")
            
            st.markdown("---")
            st.subheader("Visual Comparison")
            
            # Create a bar chart for comparison
            comparison_df = pd.DataFrame({
                'Model': ['Primary Model', 'Comparison Model'],
                'Risk Probability': [comp_res['primary']['prob'], comp_res['mock']['prob']]
            })
            
            fig_compare = px.bar(
                comparison_df,
                x='Model',
                y='Risk Probability',
                title='Risk Probability Comparison',
                color='Model',
                color_discrete_map={'Primary Model': '#4B90F9', 'Comparison Model': '#B299E5'}
            )
            fig_compare.update_layout(
                plot_bgcolor='#222E3A',
                paper_bgcolor='#222E3A',
                font_color='#E0E0E0',
                title_font_color='#E0E0E0',
                yaxis_range=[0, 1]
            )
            st.plotly_chart(fig_compare, use_container_width=True)
        # --- NEW: Educational Content Hub Page ---
    
    elif page_selection == "Educational Hub":
        st.header("❤️ CardioHealth Educational Hub")
        st.markdown("Explore key topics to better understand and manage your cardiovascular health.")
        st.markdown("---")

        topic = st.selectbox(
            "Choose a topic to learn more about:",
            options=list(EDUCATIONAL_CONTENT.keys())
        )

        if topic:
            content = EDUCATIONAL_CONTENT[topic]
            
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader(topic)
                st.write(content["summary"])

                st.markdown("#### Key Takeaways:")
                for point in content["key_points"]:
                    st.markdown(f"- {point}")

            with col2:
                # Get the local image path and convert it to Base64
                image_path = content.get("image")
                if image_path:
                    img_base64 = get_image_as_base64(image_path)
                    if img_base64:
                        st.markdown(
                            f'<div class="media-container"><img src="{img_base64}" alt="Visual for {topic}"></div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.warning(f"Image not found at path: {image_path}")
            
            st.markdown("---")
            st.subheader("Learn More (Video)")
            
            if content.get("video_url"):
                video_id = get_youtube_id(content["video_url"])
                
                if video_id:
                    embed_url = f"https://www.youtube.com/embed/{video_id}"
                    st.markdown(
                        f'<div class="media-container"><iframe src="{embed_url}" allowfullscreen></iframe></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("Could not process the video URL. Please check the link in the code.")   
def main():
    # FIXED: st.set_page_config() is now the first command in the script execution.
    st.set_page_config(
        page_title="CardioVascular Analytics Platform",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="auto"
    )

    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
        st.session_state['username'] = ""

    if st.session_state['authenticated']:
        # FIXED: Display the title here, AFTER authentication check
        st.title("❤️ CardioVascular Analytics Platform")
        st.markdown("### Advanced Cardiovascular Risk Assessment for Personalized Insights")
        
        create_streamlit_gui(st.session_state['username'])
    
    else:
        # This is the login page UI
        st.title("❤️ Welcome to CardioVascular Analytics Platform")
        st.sidebar.title("User Account")
        choice = st.sidebar.selectbox("Login / Signup", ["Login", "Sign Up"])

        if choice == "Login":
            st.sidebar.subheader("Login to Your Account")
            username = st.sidebar.text_input("Username")
            password = st.sidebar.text_input("Password", type='password')
            if st.sidebar.button("Login"):
                if username and password:
                    result = db.login_user(username, password)
                    if result:
                        st.session_state['authenticated'] = True
                        st.session_state['username'] = username
                        st.rerun()
                    else:
                        st.sidebar.error("Incorrect Username or Password")
                else:
                    st.sidebar.warning("Please enter both username and password")

        elif choice == "Sign Up":
            st.sidebar.subheader("Create a New Account")
            new_username = st.sidebar.text_input("Choose a Username")
            new_password = st.sidebar.text_input("Choose a Password", type='password')
            if st.sidebar.button("Sign Up"):
                if new_username and new_password:
                    if db.add_user(new_username, new_password):
                        st.sidebar.success("Account created successfully! Please Login.")
                    else:
                        st.sidebar.error("Username already exists. Please choose another.")
                else:
                    st.sidebar.warning("Please enter both a username and password.")
        
        st.info("Please Login or Sign Up using the sidebar to access the application.")
        st.image("signup.png", caption="Your personalized health journey starts here.")

    # This disclaimer can be at the bottom, it's fine
    st.markdown(
        """
        ---
        <p style="font-size: small; color: #6A7B8C; text-align: center;">
        Disclaimer: This assessment is for informational purposes only...
        </p>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()