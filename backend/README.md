
# CardioVascular Risk Prediction - Backend API

This directory contains the FastAPI-based backend for the application.

## Setup

1.  **Navigate to backend directory**:
    ```bash
    cd backend
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Server**:
    ```bash
    uvicorn app.main:app --reload
    ```

    The API will be available at `http://localhost:8000`.
    Interactive documentation (Swagger UI) is at `http://localhost:8000/docs`.

## Project Structure

-   `app/`: Main application code.
    -   `core/`: Database and configuration.
    -   `models/`: Pydantic schemas.
    -   `routers/`: API endpoints.
    -   `services/`: Business logic (ML, PDF, Auth).
-   `data/`: Datasets and SQLite DB.
-   `models/`: Pickle files for ML models.
-   `assets/`: Static assets (fonts, images).
