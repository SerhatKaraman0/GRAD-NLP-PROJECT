# Dockerfile
FROM python:3.11-slim

# Copy requirements.txt to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . /app

# Set the working directory
WORKDIR /app

# Ensure the data directory is copied
COPY data /app/data

EXPOSE 8080

CMD ["python3", "nlp_model.py"]