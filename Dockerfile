# Use the official Python 3.9 base image
FROM python:3.9

# Install system dependencies needed for GUI libraries (like OpenCV) and cleanup
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
    # Clean up package lists to reduce image size

# Copy requirements file first (allows Docker to cache this layer if unchanged)
COPY requirements.txt .  

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt  
# --no-cache-dir reduces image size

# Copy the entire application code into the container
COPY . /app
# Copies all files from host to /app directory in container

# Set the working directory for subsequent commands
WORKDIR /app
# All following commands will run from /app

# Inform Docker that the container listens on port 8501 (Streamlit's default)
EXPOSE 8501
# Note: This is documentation only, doesn't actually publish the port

# Command to run when container starts (launches Streamlit app)
CMD ["streamlit", "run", "frontEnd.py", "--server.address=0.0.0.0"]
# --server.address=0.0.0.0 allows external connections to the Streamlit server
