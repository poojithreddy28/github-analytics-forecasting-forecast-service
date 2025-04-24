# Use a slim Python base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Set default port for Cloud Run
EXPOSE 8081

# Run the app
CMD ["python", "app.py"]