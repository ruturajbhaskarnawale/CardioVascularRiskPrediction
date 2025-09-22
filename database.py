# database.py

import sqlite3
import hashlib

# --- Database Connection ---
conn = sqlite3.connect('user_data.db', check_same_thread=False)
c = conn.cursor()

# --- Password Hashing ---
def make_hashes(password):
    """Hashes a password using SHA256."""
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    """Checks if a password matches its hashed version."""
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

# --- User Table Functions ---
def create_user_table():
    """Creates the user table if it doesn't exist."""
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT PRIMARY KEY, password TEXT)')

def add_user(username, password):
    """Adds a new user to the user table."""
    try:
        c.execute('INSERT INTO userstable(username, password) VALUES (?,?)', (username, make_hashes(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError: # This error occurs if the username already exists
        return False

def login_user(username, password):
    """Logs in a user by verifying their credentials."""
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?', (username, make_hashes(password)))
    data = c.fetchall()
    return data

# --- Prediction History Table Functions ---
def create_prediction_table():
    """Creates the prediction history table."""
    # <<< MODIFIED: Added patient_name and patient_phone columns
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

# <<< MODIFIED: Updated function to accept and insert new patient details
def add_prediction(username, patient_name, patient_phone, timestamp, probability, risk_level, source):
    """Adds a prediction record for a specific user, including patient details."""
    c.execute(
        'INSERT INTO predictionstable(username, patient_name, patient_phone, timestamp, probability, risk_level, source) VALUES (?,?,?,?,?,?,?)',
        (username, patient_name, patient_phone, timestamp, probability, risk_level, source)
    )
    conn.commit()

def get_prediction_history(username):
    """Retrieves all prediction records for a specific user."""
    c.execute('SELECT timestamp, probability, risk_level, source FROM predictionstable WHERE username =?', (username,))
    data = c.fetchall()
    return data

# --- Initialize Database ---
def initialize_db():
    """Creates both tables."""
    create_user_table()
    create_prediction_table()

# Run initialization when the module is loaded
initialize_db()