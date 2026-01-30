# CardioVascular Risk Prediction

A sophisticated Streamlit-based web application for predicting cardiovascular disease risk using machine learning. This platform offers personalized risk assessments, bulk prediction capabilities, health resource locators, and educational content.

## Features

- **Single Prediction**: Predict cardiovascular risk for individual patients based on health metrics.
- **Bulk Prediction**: Process CSV files for batch predictions with interactive EDA visualizations.
- **Local Resources**: Find nearby hospitals and clinics in Maharashtra, India.
- **Risk Trend Analysis**: Track patient risk over time with manual entry.
- **What-If Analysis**: Simulate how lifestyle changes affect risk probabilities.
- **Model Comparison**: Compare the primary ML model against a baseline mock model.
- **Educational Hub**: informative content on heart health with videos and summaries.
- **PDF Reports**: Generate downloadable PDF reports with personalized recommendations.

## Project Structure

```
CardioVascularRiskPrediction/
├── assets/
│   ├── fonts/               # Font files for PDF generation
│   └── images/              # Images for UI and educational content
├── data/                    # Data files (CSVs, SQLite database)
├── models/                  # Trained ML models and scalers
├── Main_App.py             # Main Streamlit application entry point
├── database.py             # Database management (SQLite)
├── dataProcessing_ModelTrainning.py # Script for training the model
├── testing.py              # Script to generate sample data for testing
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher

### Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Application
To start the web application, run:
```bash
streamlit run Main_App.py
```
The app will open in your default web browser at `http://localhost:8501`.

### User Guide
- **Login/Signup**: Create an account to save your prediction history.
- **Predictions**: Navigate to "Single Prediction" or "Bulk Prediction" from the sidebar.
- **Resources**: Use "Local Resources" to find healthcare facilities (requires `india_health_facilities.csv` in `data/`).

## Data Sources
- **Training Data**: `data/cardio_train.csv` (Kaggle dataset for cardiovascular disease).
- **Healthcare Facilities**: `data/india_health_facilities.csv` (Public dataset for Maharashtra).

## Development
Toretrain the model:
```bash
python dataProcessing_ModelTrainning.py
```
To generate sample test data:
```bash
python testing.py
```

## Credits
Ruturaj Nawale - Project Developer & ML Engineer
