# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
# Assumes build context is the 'app' directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the service code
COPY SpeedLimitController /app/SpeedLimitController

# Set the working directory to the service folder
WORKDIR /app/SpeedLimitController

# Set PYTHONPATH so Python can find the modules
ENV PYTHONPATH=/app

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
