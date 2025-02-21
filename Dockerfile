FROM python:3.11-slim

# Set the working directory first
WORKDIR /app

# Copy and install dependencies first (allows Docker caching)
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt  

# Copy the rest of the application
COPY . .  

# Ensure data directory exists
RUN mkdir -p /app/data

EXPOSE 8080

CMD ["python3", "-m", "nlpmodel"]
