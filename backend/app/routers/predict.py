
from fastapi import APIRouter, HTTPException, UploadFile, File
from ..models.schemas import PredictionInput, PredictionResponse, BulkPredictionResponse
from ..services.prediction_service import prediction_service
from ..services.recommendation_engine import generate_recommendations
from ..services.history_service import history_service
from ..services.pdf_service import generate_pdf_report
from fastapi.responses import Response
import pandas as pd
import io
import datetime

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)

@router.post("/", response_model=PredictionResponse)
async def predict_single(input_data: PredictionInput, save_history: bool = False, username: str = "guest"):
    try:
        # Convert Pydantic model to dict
        user_input = input_data.model_dump()
        
        # Get Prediction
        result_text, probability = prediction_service.predict_disease(user_input)
        
        # Get Recommendations
        rec_data = generate_recommendations(user_input, probability, result_text)
        
        # Save to History if requested
        if save_history and username != "guest":
            history_service.add_prediction(
                username=username,
                patient_name=user_input.get('full_name', 'Unknown'),
                patient_phone=user_input.get('phone_number', ''),
                timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                probability=probability,
                risk_level=rec_data['risk_level'],
                source="Single Prediction API"
            )
            
        return PredictionResponse(
            result=result_text,
            probability=probability,
            risk_level=rec_data['risk_level'],
            risk_color=rec_data['risk_color'],
            recommendations=rec_data['recommendations'],
            factors=rec_data['factors'],
            screening=rec_data['screening'],
            resources=rec_data['resources']
        )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bulk", response_model=BulkPredictionResponse)
async def predict_bulk(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")
        
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        result_df = prediction_service.bulk_predict_disease(df)
        
        avg_risk = float(result_df['Prediction_Probability'].mean())
        total_records = len(result_df)
        
        # Convert DataFrame to list of dicts for JSON response
        predictions_list = result_df.to_dict(orient='records')
        
        return BulkPredictionResponse(
            total_records=total_records,
            avg_risk=avg_risk,
            predictions=predictions_list
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

@router.post("/report/pdf")
async def get_pdf_report(input_data: PredictionInput):
    """
    Generates a PDF report and returns it as a downloadable file.
    """
    try:
        user_input = input_data.model_dump()
        result_text, probability = prediction_service.predict_disease(user_input)
        rec_data = generate_recommendations(user_input, probability, result_text)
        
        pdf_bytes = generate_pdf_report(user_input, rec_data)
        
        headers = {
            'Content-Disposition': f'attachment; filename="CardioReport.pdf"'
        }
        return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
