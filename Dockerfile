# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Copy the requirements file into the container
COPY requirements.txt /app/requirements.txt

# Install system dependencies and Python packages
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && pip install --no-cache-dir -r /app/requirements.txt

# Expose port 5000 for the Flask application
EXPOSE 5005

# Define environment variable for Flask
ENV FLASK_APP=app.py

# Run Flask when the container launches
CMD ["flask", "run", "--host=0.0.0.0", "--port=5005"]

