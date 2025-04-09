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


# Database Configuration
def get_db_config():
    """Handle both Railway and local database configurations"""
    if os.getenv("DATABASE_URL"):  # Railway environment
        db_url = urllib.parse.urlparse(os.getenv("DATABASE_URL"))
        return {
            "host": db_url.hostname,
            "dbname": db_url.path[1:],
            "user": db_url.username,
            "password": db_url.password,
            "port": db_url.port,
        }
    else:  # Local development
        return {
            "host": os.getenv("DB_HOST", "db"),
            "dbname": os.getenv("DB_NAME", "mnist"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "postgres"),
            "port": os.getenv("DB_PORT", "5432"),
        }


# Define the model
class SimpleFCN(nn.Module):
    def __init__(self):
        super(SimpleFCN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Load the model
def load_model(model_path):
    model = SimpleFCN()
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


# Database connection with retry logic
def connect_to_db():
    max_retries = 5
    retry_delay = 2
    db_config = get_db_config()

    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(
                host=db_config["host"],
                dbname=db_config["dbname"],
                user=db_config["user"],
                password=db_config["password"],
                port=db_config["port"],
            )
            return conn
        except Psycopg2OpError as e:
            if attempt == max_retries - 1:
                st.error(
                    """
                Database connection failed. Please check:
                1. Database service is running
                2. Connection credentials are correct
                3. Network access is configured
                Error details: {}
                """.format(str(e))
                )
                raise
            time.sleep(retry_delay)
            st.warning(f"Retrying database connection ({attempt + 1}/{max_retries})...")


# Log prediction to PostgreSQL
def log_prediction(predicted, true_label):
    try:
        conn = connect_to_db()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO predictions (timestamp, predicted, true_label) VALUES (%s, %s, %s)",
                (datetime.now(), predicted, true_label),
            )
            conn.commit()
    except Exception as e:
        st.error(f"Error logging prediction: {e}")
    finally:
        if conn:
            conn.close()


# Fetch all predictions
def fetch_predictions():
    try:
        conn = connect_to_db()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT timestamp, predicted, true_label 
                FROM predictions 
                ORDER BY timestamp DESC
                LIMIT 100
            """)
            return cur.fetchall()
    except Exception as e:
        st.error(f"Error fetching predictions: {e}")
        return []
    finally:
        if conn:
            conn.close()


# Streamlit App
def main():
    st.title("MNIST Digit Classifier")

    # Load model
    try:
        model = load_model("mnist_cnn.pth")
    except Exception as e:
        st.error(f"Model loading failed: {e}")
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
            image = Image.fromarray(canvas_result.image_data.astype("uint8")).convert(
                "L"
            )
            image = image.resize((28, 28))
            image = transforms.ToTensor()(image).unsqueeze(0)

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
                log_prediction(predicted.item(), true_label)
                st.rerun()  # Refresh the prediction log

        except Exception as e:
            st.error(f"Prediction error: {e}")

    # Prediction history
    st.header("Recent Predictions")
    try:
        predictions = fetch_predictions()
        if predictions:
            df = pd.DataFrame(
                predictions,
                columns=["Timestamp", "Predicted", "True Label"],
            )
            st.dataframe(
                df,
                column_config={
                    "Timestamp": st.column_config.DatetimeColumn(
                        "Timestamp",
                        format="YYYY-MM-DD HH:mm:ss",
                    )
                },
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No predictions logged yet.")
    except Exception as e:
        st.error(f"Error displaying history: {e}")


if __name__ == "__main__":
    main()
