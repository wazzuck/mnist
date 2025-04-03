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


# Define the model
class SimpleFCN(nn.Module):
    def __init__(self):
        super(SimpleFCN, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Input size: 784 (28x28 flattened image)
        self.fc2 = nn.Linear(128, 64)  # Hidden layer
        self.fc3 = nn.Linear(64, 10)  # Output layer (10 classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Load the model
def load_model(model_path):
    model = SimpleFCN()
    state_dict = torch.load(
        model_path, map_location=torch.device("cpu")
    )  # Ensure model is loaded on CPU
    model.load_state_dict(state_dict)
    model.eval()
    return model


# Connect to PostgreSQL database
def connect_to_db():
    conn = psycopg2.connect(
        dbname="mnist",
        user="postgres",
        password="postgres",
        host="localhost",
        port="5432",
    )
    return conn


# Log prediction to PostgreSQL
def log_prediction(predicted, true_label):
    conn = connect_to_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions (timestamp, predicted, true_label) VALUES (%s, %s, %s)",
        (datetime.now(), predicted, true_label),
    )
    conn.commit()
    cur.close()
    conn.close()


# Fetch all predictions from PostgreSQL (exclude 'id' column)
def fetch_predictions():
    conn = connect_to_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT timestamp, predicted, true_label FROM predictions ORDER BY timestamp DESC"  # <-- No 'id' column
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


# Streamlit app
st.title("MNIST Digit Classifier")

# Load the model
model_path = "mnist_cnn.pth"  # Ensure this path is correct
model = load_model(model_path)

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
    # Preprocess the image
    image = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
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
        # Log prediction to PostgreSQL
        log_prediction(predicted.item(), true_label)
        st.write("Logged to database!")

# Display stored results in a table (without 'id' column)
st.header("Prediction Logs")
predictions = fetch_predictions()
if predictions:
    df = pd.DataFrame(
        predictions,
        columns=["Timestamp", "Predicted", "True Value"],  # <-- Custom column names
    )
    st.dataframe(df)  # <-- Displays only the 3 desired columns
else:
    st.write("No predictions logged yet.")
