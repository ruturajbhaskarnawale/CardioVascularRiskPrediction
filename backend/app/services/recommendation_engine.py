
from typing import Dict, Any

def generate_recommendations(user_input: Dict[str, Any], probability: float, result_text: str) -> Dict[str, Any]:
    """
    Generates personalized recommendations based on risk factors.
    """
    rec_data = {}
    
    # Risk Level Classification
    if probability > 0.75:
        rec_data['risk_level'] = "High Risk"
        rec_data['risk_color'] = "red"
    elif probability > 0.45:
        rec_data['risk_level'] = "Moderate Risk"
        rec_data['risk_color'] = "orange"
    else:
        rec_data['risk_level'] = "Low Risk"
        rec_data['risk_color'] = "green"

    rec_data['probability'] = f"{probability*100:.1f}%"
    rec_data['prob_value'] = probability
    rec_data['prediction_result'] = result_text

    # Analyze Risk Factors
    factors = []
    
    # Check BP
    if user_input['ap_hi'] >= 130 or user_input['ap_lo'] >= 80:
        factors.append("- **High Blood Pressure**: Your BP readings are elevated.")
    
    # Check BMI
    bmi = user_input['weight'] / (user_input['height'] / 100)**2
    if bmi >= 25:
        factors.append(f"- **Overweight/Obese**: Your BMI is {bmi:.1f}, which is above the healthy range.")
    
    # Check Cholesterol (Assuming 1=Normal, 2=Above, 3=Well Above)
    # Map input if it's string to int
    if isinstance(user_input['cholesterol'], str):
         chol_map = {"Normal (1)": 1, "Above Normal (2)": 2, "Well Above (3)": 3}
         chol_val = chol_map.get(user_input['cholesterol'], 1)
    else:
         chol_val = user_input['cholesterol']
         
    if chol_val > 1:
        factors.append("- **Cholesterol**: Levels are above normal.")

    # Check Glue (Assuming 1=Normal, 2=Above, 3=Well Above)
    if isinstance(user_input['gluc'], str):
         gluc_map = {"Normal (1)": 1, "Above Normal (2)": 2, "Well Above (3)": 3}
         gluc_val = gluc_map.get(user_input['gluc'], 1)
    else:
         gluc_val = user_input['gluc']
         
    if gluc_val > 1:
        factors.append("- **Glucose**: Levels are above normal.")

    # Check Smoke
    if user_input.get('smoke') == 1 or user_input.get('smoke') == "Smoker":
        factors.append("- **Smoking**: Active smoking is a major risk factor.")

    # Check Alcohol
    if user_input.get('alco') in [1, 2] or user_input.get('alco') in ["Moderate Drinker", "Heavy Drinker"]:
         factors.append("- **Alcohol Consumption**: Alcohol intake contributes to risk.")

    # Check Activity
    if user_input.get('active') == 0 or user_input.get('active') == "Sedentary":
         factors.append("- **Physical Inactivity**: Lack of exercise increases risk.")

    if not factors:
        factors.append("- **None**: No major modifiable risk factors identified in this simple screening.")

    rec_data['factors'] = "\n".join(factors)

    # Recommendations
    recs = []
    if "Blood Pressure" in rec_data['factors']:
        recs.append("- **Manage BP**: Reduce sodium intake, exercise regularly, and consult a doctor.")
    if "Cholesterol" in rec_data['factors']:
         recs.append("- **Lower Cholesterol**: Eat more fiber (oats, fruits), avoid trans fats, and stay active.")
    if "Overweight" in rec_data['factors']:
         recs.append("- **Weight Management**: Aim for a gradual weight loss of 5-10% using a balanced diet.")
    if "Smoking" in rec_data['factors']:
         recs.append("- **Quit Smoking**: Seek support to stop smoking immediately.")
    if "Inactivity" in rec_data['factors']:
         recs.append("- **Exercise**: Aim for at least 150 mins of moderate activity per week.")
    
    if not recs:
         recs.append("- **Maintain Healthy Habits**: Keep up the good work with your diet and activity levels!")
         
    rec_data['recommendations'] = "\n".join(recs)
    
    # Screening and Resources
    screen = []
    if rec_data['risk_level'] == "High Risk":
        screen.append("- **Immediate Action**: Consult a cardiologist within the next week.")
        screen.append("- **Tests**: Lipid profile, ECG, and blood sugar tests recommended.")
    elif rec_data['risk_level'] == "Moderate Risk":
        screen.append("- **Monitoring**: Check BP weekly. Consult a GP within a month.")
    else:
        screen.append("- **Routine Checkup**: continue annual checkups.")
        
    rec_data['screening'] = "\n".join(screen)
    
    rec_data['resources'] = "Check the 'Educational Hub' in the app for videos on BP and Diet."

    return rec_data
