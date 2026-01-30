
import hashlib
import sqlite3
from ..core.database import get_db_connection

class AuthService:
    def _make_hashes(self, password: str) -> str:
        """Hashes a password using SHA256 (Compatibility Mode)."""
        return hashlib.sha256(str.encode(password)).hexdigest()

    def check_hashes(self, password: str, hashed_text: str) -> bool:
        """Checks if a password matches its hashed version."""
        if self._make_hashes(password) == hashed_text:
            return True
        return False

    def create_user(self, username, password):
        """Adds a new user to the user table."""
        conn = get_db_connection()
        c = conn.cursor()
        try:
            c.execute('INSERT INTO userstable(username, password) VALUES (?,?)', 
                      (username, self._make_hashes(password)))
            conn.commit()
            return True
        except sqlite3.IntegrityError: # Username already exists
            return False
        finally:
            conn.close()

    def login_user(self, username, password):
        """Logs in a user by verifying their credentials."""
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT * FROM userstable WHERE username =? AND password = ?', 
                  (username, self._make_hashes(password)))
        data = c.fetchall()
        conn.close()
        return data

auth_service = AuthService()
