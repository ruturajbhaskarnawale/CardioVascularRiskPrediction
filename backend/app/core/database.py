
import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "data", "user_data.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries (like)
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # User Table
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT PRIMARY KEY, password TEXT)')
    
    # Prediction Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictionstable(
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            patient_name TEXT, 
            patient_phone TEXT,
            timestamp TEXT,
            probability REAL,
            risk_level TEXT,
            source TEXT,
            FOREIGN KEY(username) REFERENCES userstable(username)
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")

# Initialize on module load or manually called
init_db()
