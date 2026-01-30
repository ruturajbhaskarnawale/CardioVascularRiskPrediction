
import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

# Define paths relative to the backend root (assuming running from backend/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cardio_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

class PredictionService:
    def __init__(self):
        self.best_model = None
        self.scaler = None
        self.numerical_cols = ['age', 'height', 'ap_hi', 'ap_lo', 'bmi']
        self._load_models()

    def _load_models(self):
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
                self.best_model = joblib.load(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                print("Models loaded successfully.")
            else:
                print(f"Error: Model files not found at {MODEL_PATH} or {SCALER_PATH}")
        except Exception as e:
             print(f"Error loading models: {e}")

    def predict_disease(self, user_input: Dict[str, Any]) -> Tuple[str, float]:
        """
        Prepares user input, scales numerical features, and predicts cardiovascular disease risk.
        """
        if not self.best_model or not self.scaler:
             raise RuntimeError("Models not loaded correctly.")

        # Calculate derived features
        bmi = user_input['weight'] / (user_input['height'] / 100)**2
        # Input age is in years, convert to days for model as previously trained?
        # Checking Main_App.py: age_in_days = user_input['age'] * 365.25
        age_in_days = user_input['age'] * 365.25

        data_for_prediction = {
            'age': age_in_days,
            'gender': 1 if user_input['gender'] == 'Female' else 0, # Assuming 'Male'/'Female' string input
            'height': user_input['height'],
            'ap_hi': user_input['ap_hi'],
            'ap_lo': user_input['ap_lo'],
            'cholesterol': user_input['cholesterol'], # Assuming mapped integer 1,2,3
            'gluc': user_input['gluc'],             # Assuming mapped integer 1,2,3
            'smoke': user_input['smoke'],           # Assuming 0/1
            'alco': user_input['alco'],             # Assuming 0/1 (mapped from levels)
            'active': user_input['active'],         # Assuming 0/1 (mapped from levels)
            'bmi': bmi
        }

        model_features = [
            'age', 'gender', 'height', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
            'smoke', 'alco', 'active', 'bmi'
        ]

        input_df = pd.DataFrame([data_for_prediction], columns=model_features)
        df_scaled = input_df.copy()
        df_scaled[self.numerical_cols] = self.scaler.transform(df_scaled[self.numerical_cols])

        prob = self.best_model.predict_proba(df_scaled)[:, 1][0]
        prediction = self.best_model.predict(df_scaled)[0]
        result = "Cardiovascular Disease" if prediction == 1 else "No Cardiovascular Disease"

        return result, float(prob)

    def bulk_predict_disease(self, df_bulk: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a DataFrame of user inputs for bulk prediction.
        """
        if not self.best_model or not self.scaler:
             raise RuntimeError("Models not loaded correctly.")
             
        df_processed = df_bulk.copy()

        # Mappings (logic from Main_App.py)
        gender_map_csv = {"Male": 0, "Female": 1, 0:0, 1:1}
        cholesterol_map_csv = {"Normal (1)": 1, "Above Normal (2)": 2, "Well Above (3)": 3, 1:1, 2:2, 3:3}
        gluc_map_csv = {"Normal (1)": 1, "Above Normal (2)": 2, "Well Above (3)": 3, 1:1, 2:2, 3:3}
        smoke_map_csv = {"Non-smoker": 0, "Smoker": 1, 0:0, 1:1}
        alco_map_csv = {"Non-drinker": 0, "Moderate Drinker": 1, "Heavy Drinker": 2, 0:0, 1:1, 2:2}
        active_map_csv = {"Sedentary": 0, "Moderately Active": 1, "Very Active": 2, 0:0, 1:1, 2:2}

        required_cols = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        
        # Validation checks can be done at API level or here
        
        df_processed['bmi'] = df_processed['weight'] / (df_processed['height'] / 100)**2
        df_processed['age'] = df_processed['age'] * 365.25

        df_processed['gender'] = df_processed['gender'].map(gender_map_csv).fillna(0)
        df_processed['cholesterol'] = df_processed['cholesterol'].map(cholesterol_map_csv).fillna(1)
        df_processed['gluc'] = df_processed['gluc'].map(gluc_map_csv).fillna(1)
        df_processed['smoke'] = df_processed['smoke'].map(smoke_map_csv).fillna(0)
        df_processed['alco'] = df_processed['alco'].map(alco_map_csv).fillna(0)
        df_processed['active'] = df_processed['active'].map(active_map_csv).fillna(1)

        model_features_order = [
            'age', 'gender', 'height', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
            'smoke', 'alco', 'active', 'bmi'
        ]
        
        # Ensure columns exist and are ordered
        df_input = df_processed[model_features_order].copy()
        
        df_input[self.numerical_cols] = self.scaler.transform(df_input[self.numerical_cols])

        probabilities = self.best_model.predict_proba(df_input)[:, 1]
        predictions = self.best_model.predict(df_input)

        df_bulk['Predicted_Cardio_Disease'] = np.where(predictions == 1, 'Yes', 'No')
        df_bulk['Prediction_Probability'] = probabilities
        
        return df_bulk

# Instantiate a single global service
prediction_service = PredictionService()
