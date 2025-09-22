import pandas as pd
import numpy as np

def create_sample_csv(filename="sample_cardio_data.csv"):
    """
    Creates a sample CSV file with dummy cardiovascular health data
    for testing the bulk prediction feature.

    Args:
        filename (str): The name of the CSV file to create.
    """
    data = {
        'age': [35, 50, 62, 45, 70, 30, 55, 40, 68, 48], # Age in years
        'gender': ['Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'], # String representation
        'height': [165, 178, 160, 170, 155, 180, 168, 172, 163, 175], # cm
        'weight': [68, 85, 72, 75, 60, 90, 70, 78, 65, 82], # kg
        'ap_hi': [120, 140, 160, 130, 150, 110, 135, 125, 155, 128], # Systolic BP
        'ap_lo': [80, 90, 100, 85, 95, 70, 88, 82, 98, 83], # Diastolic BP
        'cholesterol': ['Normal (1)', 'Above Normal (2)', 'Well Above (3)', 'Normal (1)', 'Above Normal (2)',
                        'Normal (1)', 'Above Normal (2)', 'Normal (1)', 'Well Above (3)', 'Normal (1)'], # String representation
        'gluc': ['Normal (1)', 'Normal (1)', 'Above Normal (2)', 'Normal (1)', 'Well Above (3)',
                 'Normal (1)', 'Normal (1)', 'Above Normal (2)', 'Normal (1)', 'Normal (1)'], # String representation
        'smoke': ['Non-smoker', 'Smoker', 'Non-smoker', 'Non-smoker', 'Smoker',
                  'Non-smoker', 'Non-smoker', 'Smoker', 'Non-smoker', 'Non-smoker'], # String representation
        'alco': ['Non-drinker', 'Moderate Drinker', 'Non-drinker', 'Non-drinker', 'Heavy Drinker',
                 'Non-drinker', 'Moderate Drinker', 'Non-drinker', 'Heavy Drinker', 'Non-drinker'], # String representation
        'active': ['Moderately Active', 'Sedentary', 'Very Active', 'Moderately Active', 'Sedentary',
                   'Very Active', 'Moderately Active', 'Sedentary', 'Very Active', 'Moderately Active'], # String representation
        'stress': ['Low', 'High', 'Moderate', 'Low', 'High', 
                   'Low', 'Moderate', 'High', 'Moderate', 'Low'] # Optional for recommendations
    }

    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Sample CSV '{filename}' created successfully with {len(df)} rows.")

if __name__ == "__main__":
    create_sample_csv()
