
from ..core.database import get_db_connection
from typing import Optional

class HistoryService:
    def add_prediction(
        self, 
        username: str, 
        patient_name: str, 
        patient_phone: str, 
        timestamp: str, 
        probability: float, 
        risk_level: str, 
        source: str
    ):
        """Adds a prediction record for a specific user."""
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            'INSERT INTO predictionstable(username, patient_name, patient_phone, timestamp, probability, risk_level, source) VALUES (?,?,?,?,?,?,?)',
            (username, patient_name, patient_phone, timestamp, probability, risk_level, source)
        )
        conn.commit()
        conn.close()

    def get_history(self, username: str):
        """Retrieves all prediction records for a specific user."""
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT timestamp, probability, risk_level, source FROM predictionstable WHERE username =?', (username,))
        data = c.fetchall()
        # Convert row objects to dicts
        result = [dict(row) for row in data]
        conn.close()
        return result

history_service = HistoryService()
