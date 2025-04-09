# Import required libraries
import streamlit as st  # Web app framework
import torch  # PyTorch for deep learning
import torch.nn as nn  # Neural network modules
from torchvision import transforms  # Image transformations
from PIL import Image  # Image processing
from streamlit_drawable_canvas import st_canvas  # Drawing canvas component
import psycopg2  # PostgreSQL database adapter
from datetime import datetime  # Timestamp handling
import pandas as pd  # Data manipulation and display


# Define the neural network model architecture
class SimpleFCN(nn.Module):
    def __init__(self):
        super(SimpleFCN, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Input layer (28x28=784 pixels to 128 neurons)
        self.fc2 = nn.Linear(128, 64)  # Hidden layer (128 to 64 neurons)
        self.fc3 = nn.Linear(64, 10)  # Output layer (64 to 10 classes for digits 0-9)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input image tensor
        x = torch.relu(self.fc1(x))  # ReLU activation for first layer
        x = torch.relu(self.fc2(x))  # ReLU activation for second layer
        x = self.fc3(x)  # Final output layer (no activation)
        return x


# Load pre-trained model weights
def load_model(model_path):
    model = SimpleFCN()  # Initialize model instance
    state_dict = torch.load(
        model_path,
        map_location=torch.device("cpu"),  # Load weights onto CPU
    )
    model.load_state_dict(state_dict)  # Apply weights to model
    model.eval()  # Set model to evaluation mode
    return model


# Database connection function
def connect_to_db():
    conn = psycopg2.connect(
        dbname="mnist",  # Database name
        user="postgres",  # Database username
        password="postgres",  # Database password
        host="db",  # Service name from docker-compose
        port="5432",  # Default PostgreSQL port
    )
    return conn


# Log prediction results to database
def log_prediction(predicted, true_label):
    conn = connect_to_db()
    cur = conn.cursor()
    # SQL query to insert prediction record
    cur.execute(
        "INSERT INTO predictions (timestamp, predicted, true_label) VALUES (%s, %s, %s)",
        (datetime.now(), predicted, true_label),  # Current time and values
    )
    conn.commit()  # Save changes
    cur.close()
    conn.close()


# Retrieve prediction history from database
def fetch_predictions():
    conn = connect_to_db()
    cur = conn.cursor()
    # SQL query to get recent predictions (without ID column)
    cur.execute(
        "SELECT timestamp, predicted, true_label FROM predictions ORDER BY timestamp DESC"
    )
    rows = cur.fetchall()  # Get all results
    cur.close()
    conn.close()
    return rows


# --- Streamlit Application UI ---
st.title("MNIST Digit Classifier")  # App title

# Load the pre-trained model
model_path = "mnist_cnn.pth"  # Model weights file
model = load_model(model_path)  # Initialize model

# Create drawing canvas component
canvas_result = st_canvas(
    fill_color="#000000",  # Black background
    stroke_width=10,  # Brush thickness
    stroke_color="#FFFFFF",  # White drawing color
    background_color="#000000",  # Canvas background
    width=280,  # Canvas width (px)
    height=280,  # Canvas height (px)
    drawing_mode="freedraw",  # Free drawing mode
    key="canvas",  # Unique identifier
)

# Process when user draws something
if canvas_result.image_data is not None:
    # Convert canvas data to grayscale image
    image = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
    image = image.resize((28, 28))  # Resize to MNIST dimensions
    image = transforms.ToTensor()(image).unsqueeze(0)  # Convert to tensor format

    # Make prediction
    with torch.no_grad():  # Disable gradient calculation
        output = model(image)
        probabilities = torch.softmax(output, dim=1)  # Convert to probabilities
        confidence, predicted = torch.max(probabilities, 1)  # Get top prediction

    # Display results
    st.write(f"Prediction: {predicted.item()}")  # Predicted digit
    st.write(f"Confidence: {confidence.item():.2f}")  # Prediction confidence (0-1)

    # User feedback input
    true_label = st.number_input(
        "Enter the true label (0-9):", min_value=0, max_value=9, step=1
    )

    # Submit button action
    if st.button("Submit"):
        log_prediction(predicted.item(), true_label)  # Save to database
        st.write("Logged to database!")

# Display prediction history
st.header("Prediction Logs")
predictions = fetch_predictions()
if predictions:
    # Create formatted dataframe for display
    df = pd.DataFrame(
        predictions,
        columns=["Timestamp", "Predicted", "True Value"],  # Column headers
    )
    st.dataframe(df)  # Interactive table display
else:
    st.write("No predictions logged yet.")  # Empty state message
