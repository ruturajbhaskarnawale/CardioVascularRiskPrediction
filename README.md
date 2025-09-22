# CardioVascularRiskPrediction
CardioVascular Analytics Platform
Project Overview
The CardioVascular Analytics Platform is an advanced machine learning application designed to assess cardiovascular disease risk using personalized health metrics. Built with Streamlit, this platform offers an interactive, user-friendly interface for both individual and bulk predictions, along with features like local resource lookup, risk trend analysis, what-if scenario analysis, model comparison, and an educational hub. The application leverages a pre-trained machine learning model (cardio_model.pkl) and a scaler (scaler.pkl) to predict cardiovascular disease risk based on inputs such as age, blood pressure, cholesterol, and lifestyle factors.
The platform is designed for healthcare professionals, researchers, or individuals interested in cardiovascular health, providing actionable insights, personalized recommendations, and downloadable PDF reports. It also includes geospatial analysis to locate nearby healthcare resources and an educational hub with curated content to enhance heart health awareness.

Features

Single Prediction: Allows users to input individual health metrics to predict cardiovascular disease risk, accompanied by personalized recommendations and a downloadable PDF report.
Bulk Prediction (CSV): Processes CSV files containing multiple patient records for batch predictions, with interactive exploratory data analysis (EDA) visualizations using Plotly.
Local Resources: Finds nearby healthcare facilities in Maharashtra, India, using a public dataset (india_health_facilities.csv) and displays them on an interactive map.
Risk Trend Analysis: Tracks historical risk predictions over time with interactive line charts to visualize risk progression.
What-If Scenario Analysis: Enables users to simulate changes in health parameters to see their impact on predicted risk, with visual comparisons to a baseline.
Model Comparison: Compares predictions from the primary machine learning model with a mock model to demonstrate performance differences.
Educational Hub: Provides curated educational content on cardiovascular health topics, including summaries, key takeaways, images, and YouTube videos.
User Authentication: Includes a login and signup system with a SQLite database to manage user accounts and store prediction history.


Prerequisites
To run this project locally, ensure you have the following installed:

Python 3.8+
pip (Python package manager)
Required Python libraries (listed in requirements.txt)


Installation

Clone the Repository:
git clone https://github.com/yourusername/cardiovascular-analytics-platform.git
cd cardiovascular-analytics-platform


Set Up a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Download Required Files:

Ensure the following files are in the project directory:
cardio_model.pkl: Pre-trained machine learning model for predictions.
scaler.pkl: Scaler for preprocessing numerical features.
india_health_facilities.csv: Dataset for local resource lookup (available from data.gov.in).
cardio_train.csv: Original training dataset (for reference or retraining).
sample_cardio_data.csv: Sample CSV for testing bulk predictions.
Educational hub images in the educationalHub_image/ folder (bloodPressure.jpg, cholestrol.jpg, dashDiet.jpg, diabetes.jpg, physicalActivity.jpg, stress.jpg).


Note: Update image paths in the Main_App.py if running on a different system.


Set Up SQLite Database:

The application uses a SQLite database (user_data.db) for user authentication and prediction history.
The database is automatically created when the application runs for the first time, using the database.py module.


Fonts for PDF Generation:

Download DejaVuSans font files (DejaVuSans.ttf, DejaVuSans-Bold.ttf, etc.) and place them in the project directory for PDF report generation.




Running the Application

Start the Streamlit Server:
streamlit run Main_App.py


Access the Application:

Open a web browser and navigate to http://localhost:8501.
Sign up or log in using the sidebar to access the full application.




Project Structure
cardiovascular-analytics-platform/
├── Main_App.py                     # Main Streamlit application script
├── database.py                     # SQLite database management for user authentication
├── dataProcessing_ModelTrainning.py # Data preprocessing and model training script
├── cardio_model.pkl                 # Pre-trained machine learning model
├── scaler.pkl                       # Scaler for preprocessing
├── cardio_train.csv                 # Original training dataset
├── sample_cardio_data.csv           # Sample CSV for testing bulk predictions
├── india_health_facilities.csv      # Dataset for local resource lookup
├── educationalHub_image/            # Folder containing images for the educational hub
│   ├── bloodPressure.jpg
│   ├── cholestrol.jpg
│   ├── dashDiet.jpg
│   ├── diabetes.jpg
│   ├── physicalActivity.jpg
│   └── stress.jpg
├── Figure_1.png                     # Sample image (possibly for documentation)
├── signup.png                       # Image for the login page
├── _RUTURAJ_NAWALE_C25665.docx      # Project documentation (Word file)
├── RUTURAJ_NAWALE_C25665.docx       # Additional project documentation
├── testing.py                       # Test script (if any)
├── user_data.db                     # SQLite database (auto-generated)
├── __pycache__/                     # Python cache files (auto-generated)
└── DejaVuSans-*.ttf                 # Font files for PDF generation


Usage Instructions

Login/Signup:

Use the sidebar to create an account or log in. User credentials are stored securely in a SQLite database.


Single Prediction:

Enter patient details (e.g., age, height, weight, blood pressure, cholesterol) and click "Analyze Cardiovascular Risk".
View the predicted risk, personalized recommendations, and download a PDF report.


Bulk Prediction:

Upload a CSV file with the required columns (see app for details).
View interactive EDA plots and download the prediction results as a CSV.


Local Resources:

Select a city in Maharashtra and a resource type (e.g., Hospital) to find nearby facilities.
Results are displayed in a table and on an interactive map.


Risk Trend Analysis:

Add past data points to track risk over time.
Visualize trends with an interactive Plotly chart.


What-If Scenario:

Set a baseline prediction, then adjust parameters to see how changes affect risk.
Compare baseline and hypothetical risks with a bar chart.


Model Comparison:

Enter patient data to compare predictions from the primary model and a mock model.
View results and a visual comparison of risk probabilities.


Educational Hub:

Select a topic to view summaries, key takeaways, images, and embedded YouTube videos.




Model Training
The machine learning model was trained using the dataProcessing_ModelTrainning.py script:

Dataset: Cardiovascular disease training data (cardio_train.csv).
Preprocessing:
Outlier removal using IQR method.
Feature engineering: BMI calculation, age conversion to years.
Scaling numerical features with StandardScaler.
Handling class imbalance with SMOTE.


Models Evaluated: Logistic Regression, KNN, Random Forest (with GridSearchCV hyperparameter tuning).
Best Model: Random Forest (selected based on highest accuracy and ROC-AUC score).
Output: Saved as cardio_model.pkl and scaler.pkl.

To retrain the model:
python dataProcessing_ModelTrainning.py


Dependencies
The project requires the following Python packages (included in requirements.txt):
streamlit
pandas
joblib
numpy
matplotlib
plotly
fpdf
base64
sqlite3
scikit-learn
imbalanced-learn
seaborn

To generate requirements.txt:
pip freeze > requirements.txt


Notes

Model Files: Ensure cardio_model.pkl and scaler.pkl are present in the project directory, as they are critical for predictions.
Dataset: The india_health_facilities.csv file must be downloaded from data.gov.in for the Local Resources feature to work.
Image Paths: Update the paths in the EDUCATIONAL_CONTENT dictionary in Main_App.py to match your local file system if running on a different machine.
Database: The SQLite database (user_data.db) is created automatically and stores user credentials and prediction history.
Security: Passwords are hashed using SHA256 before storage. For production use, consider more robust security measures.
Fonts: DejaVuSans TTF files are required for PDF generation with international character support.


Limitations

The application assumes the presence of a pre-trained model and scaler.
The Local Resources feature is limited to Maharashtra, India, due to the dataset used.
Image paths in the Educational Hub are hardcoded and need adjustment for different systems.
The prediction history is session-based and resets on app reload (not persisted unless stored in the database).
The mock model in the comparison feature is simplified for demonstration purposes.


Future Improvements

Add persistent storage for prediction history across sessions.
Expand the Local Resources feature to include more regions or real-time API-based lookups.
Integrate additional machine learning models for comparison.
Enhance the Educational Hub with more interactive content or quizzes.
Add support for multi-language content in the Educational Hub.
Implement email notifications for high-risk predictions.
Add data export functionality for compliance (e.g., HIPAA/GDPR).


Disclaimer
This application is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for health concerns.

Contributors

Ruturaj Nawale - Project Developer & ML Engineer


License
This project is licensed under the MIT License. See the LICENSE file for details.

Contact
For questions or contributions, please reach out via the repository issues or contact Ruturaj Nawale (refer to project documentation files for contact details).
