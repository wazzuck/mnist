#!/home/neville/anaconda3/bin/python

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas
import psycopg2
from datetime import datetime
import pandas as pd
import os
import time
import urllib.parse
from psycopg2 import OperationalError as Psycopg2OpError
from typing import Optional, List, Tuple


# Database Configuration
def get_db_config() -> dict:
    """Get database configuration from environment variables"""
    if os.getenv("DATABASE_URL"):  # Railway environment
        db_url = urllib.parse.urlparse(os.getenv("DATABASE_URL"))
        return {
            "host": db_url.hostname,
            "dbname": db_url.path[1:],  # Remove leading slash
            "user": db_url.username,
            "password": db_url.password,
            "port": db_url.port,
            "sslmode": "require",
        }
    else:  # Local development
        return {
            "host": os.getenv("DB_HOST", "localhost"),
            "dbname": os.getenv("DB_NAME", "mnist"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "postgres"),
            "port": os.getenv("DB_PORT", "5432"),
            "sslmode": "disable",
        }


# Neural Network Model
class SimpleFCN(nn.Module):
    def __init__(self):
        super(SimpleFCN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_model(model_path: str) -> SimpleFCN:
    """Load trained model weights"""
    model = SimpleFCN()
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


# Database Connection
def connect_to_db() -> Optional[psycopg2.extensions.connection]:
    """Establish database connection with retry logic"""
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
                st.error(f"Database connection failed after {max_retries} attempts")
                st.error(f"Error details: {str(e)}")
                return None
            time.sleep(retry_delay)
    return None


def log_prediction(predicted: int, true_label: int) -> bool:
    """Log a prediction to the database"""
    conn = None
    try:
        conn = connect_to_db()
        if not conn:
            return False

        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO predictions (timestamp, predicted, true_label) VALUES (%s, %s, %s)",
                (datetime.now(), predicted, true_label),
            )
            conn.commit()
            return True
    except Exception as e:
        st.error(f"Error logging prediction: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()


def fetch_predictions() -> List[Tuple[datetime, int, int]]:
    """Fetch prediction history from database"""
    conn = None
    try:
        conn = connect_to_db()
        if not conn:
            return []

        with conn.cursor() as cur:
            cur.execute("""
                SELECT timestamp, predicted, true_label 
                FROM predictions 
                ORDER BY timestamp DESC
                LIMIT 100
            """)
            return cur.fetchall()
    except Exception as e:
        st.error(f"Error fetching predictions: {str(e)}")
        return []
    finally:
        if conn:
            conn.close()


# Streamlit UI
def main():
    st.title("MNIST Digit Classifier")

    # Initialize model
    try:
        model = load_model("mnist_cnn.pth")
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

    # Drawing canvas
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=10,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    # Prediction logic
    if canvas_result.image_data is not None:
        try:
            # Preprocess image
            image = Image.fromarray(canvas_result.image_data.astype("uint8")).convert(
                "L"
            )
            image = image.resize((28, 28))
            image = transforms.ToTensor()(image).unsqueeze(0)

            # Make prediction
            with torch.no_grad():
                output = model(image)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            st.success(
                f"Prediction: {predicted.item()} (Confidence: {confidence.item():.2f})"
            )

            # User feedback
            true_label = st.number_input(
                "Enter the true label (0-9):", min_value=0, max_value=9, step=1
            )

            if st.button("Submit Prediction"):
                if log_prediction(predicted.item(), true_label):
                    st.success("Prediction logged successfully!")
                else:
                    st.error("Failed to log prediction")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

    # Display prediction history
    st.header("Prediction History")
    predictions = fetch_predictions()

    if predictions:
        df = pd.DataFrame(predictions, columns=["Timestamp", "Predicted", "True Label"])
        st.dataframe(
            df.style.format({"Timestamp": lambda x: x.strftime("%Y-%m-%d %H:%M:%S")}),
            height=400,
        )
    else:
        st.info("No predictions logged yet")


if __name__ == "__main__":
    main()
