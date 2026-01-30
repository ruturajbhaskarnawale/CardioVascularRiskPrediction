
# Cardio Health Risk Predictor Pro

A modern, full-stack application for assessing cardiovascular disease risk using Machine Learning.

## Overview
This project uses a **FastAPI** backend to serve predictive models and a **Next.js** frontend for a premium user experience. It provides:
-   **Risk Assessment**: Single-patient prediction using ML.
-   **Bulk Analysis**: CSV upload for batch processing.
-   **PDF Reports**: Downloadable health reports.
-   **Resource Finder**: Location-based search for hospitals (Maharashtra).
-   **Education Hub**: Curated videos and articles.

## Quick Start

### Prerequisites
-   Python 3.8+
-   Node.js 18+

### One-Click Start (Windows)
Double-click `start_project.bat` to launch both backend and frontend servers.

### Manual Setup
**1. Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**2. Frontend**
```bash
cd frontend
npm install
npm run dev
```

## Architecture

-   **Backend**: `backend/` - Python, FastAPI, Pandas, Scikit-learn.
-   **Frontend**: `frontend/` - TypeScript, Next.js, Tailwind CSS, Shadcn UI.
-   **Legacy**: `legacy/` - Old Streamlit application files.

## Credits
Built by Ruturaj Bhaskar Nawale using the Gemini Agentic Assistant.
