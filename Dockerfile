FROM python:3.9

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    postgresql \
    postgresql-client
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

# Run Postgres
CMD ["pg_ctl","start","-D","pg"]

# Run Streaml
CMD ["streamlit", "run", "frontEnd.py"]
