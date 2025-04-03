FROM python:3.9

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . /app
WORKDIR /app

# Expose Streamlit port
EXPOSE 8501
EXPOSE 8080

# Use the official PostgreSQL image
FROM postgres:latest

# Expose PostgreSQL port
EXPOSE 5432

# Run Postgres
RUN postgres -D pg &

# Run Streamlit
CMD ["streamlit", "run", "frontEnd.py"]
