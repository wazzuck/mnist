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
from psycopg2 import OperationalError as Psycopg2OpError


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


# Connect to PostgreSQL database with retry logic
def connect_to_db():
    max_retries = 5
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(
                dbname=os.getenv("DB_NAME", "mnist"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", "postgres"),
                host=os.getenv("DB_HOST", "db"),  # Defaults to 'db' for Docker
                port=os.getenv("DB_PORT", "5432"),
            )
            return conn
        except Psycopg2OpError as e:
            if attempt == max_retries - 1:
                st.error(f"Failed to connect to database after {max_retries} attempts")
                raise e
            time.sleep(retry_delay)
            st.warning(
                f"Database connection failed (attempt {attempt + 1}/{max_retries}), retrying..."
            )


# Log prediction to PostgreSQL
def log_prediction(predicted, true_label):
    try:
        conn = connect_to_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO predictions (timestamp, predicted, true_label) VALUES (%s, %s, %s)",
            (datetime.now(), predicted, true_label),
        )
        conn.commit()
    except Exception as e:
        st.error(f"Error logging prediction: {e}")
    finally:
        if "conn" in locals():
            cur.close()
            conn.close()


# Fetch all predictions with error handling
def fetch_predictions():
    try:
        conn = connect_to_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT timestamp, predicted, true_label FROM predictions ORDER BY timestamp DESC"
        )
        return cur.fetchall()
    except Exception as e:
        st.error(f"Error fetching predictions: {e}")
        return []
    finally:
        if "conn" in locals():
            cur.close()
            conn.close()


# Streamlit app
def main():
    st.title("MNIST Digit Classifier")

    # Load the model
    model_path = "mnist_cnn.pth"
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    # Canvas for drawing
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

    if canvas_result.image_data is not None:
        try:
            # Preprocess the image
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

            st.write(f"Prediction: {predicted.item()}")
            st.write(f"Confidence: {confidence.item():.2f}")

            # User feedback
            true_label = st.number_input(
                "Enter the true label (0-9):", min_value=0, max_value=9, step=1
            )

            if st.button("Submit"):
                log_prediction(predicted.item(), true_label)
                st.success("Logged to database!")

        except Exception as e:
            st.error(f"Prediction error: {e}")

    # Display stored results
    st.header("Prediction Logs")
    try:
        predictions = fetch_predictions()
        if predictions:
            df = pd.DataFrame(
                predictions,
                columns=["Timestamp", "Predicted", "True Value"],
            )
            st.dataframe(df)
        else:
            st.info("No predictions logged yet.")
    except Exception as e:
        st.error(f"Error displaying logs: {e}")


if __name__ == "__main__":
    main()
