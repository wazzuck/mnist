import os
import psycopg2
from urllib.parse import urlparse
from psycopg2 import OperationalError as Psycopg2OpError


def get_db_config():
    """Handle both Railway and local development configurations"""
    if os.getenv("DATABASE_URL"):  # Railway environment
        db_url = urlparse(os.getenv("DATABASE_URL"))
        return {
            "host": db_url.hostname,
            "dbname": db_url.path[1:],  # Remove leading slash
            "user": db_url.username,
            "password": db_url.password,
            "port": db_url.port,
            "sslmode": "require",  # Essential for Railway
        }
    else:  # Local development
        return {
            "host": os.getenv("DB_HOST", "localhost"),  # Not 'db'
            "dbname": os.getenv("DB_NAME", "mnist"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "postgres"),
            "port": os.getenv("DB_PORT", "5432"),
            "sslmode": "disable",  # SSL not needed locally
        }


def connect_to_db():
    """Establish database connection with proper error handling"""
    max_retries = 3
    retry_delay = 1
    db_config = get_db_config()

    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(
                host=db_config["host"],
                dbname=db_config["dbname"],
                user=db_config["user"],
                password=db_config["password"],
                port=db_config["port"],
                sslmode=db_config["sslmode"],
            )
            return conn
        except Psycopg2OpError as e:
            if attempt == max_retries - 1:
                raise Exception(
                    f"Failed to connect after {max_retries} attempts: {str(e)}"
                )
            time.sleep(retry_delay)

    return None  # Explicit return if all retries fail


def fetch_predictions():
    """Safe prediction fetching with proper resource cleanup"""
    conn = None
    try:
        conn = connect_to_db()
        if not conn:
            return []

        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT timestamp, predicted, true_label 
                FROM predictions 
                ORDER BY timestamp DESC
                LIMIT 100
            """
            )
            return cursor.fetchall()
    except Exception as e:
        print(f"Database error: {str(e)}")
        return []
    finally:
        if conn:
            conn.close()
