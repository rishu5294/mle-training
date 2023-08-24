# Use an official Python runtime as the base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the contents of the src directory into the container
COPY src/ .

# Expose the port the application runs on
EXPOSE 8080

# Install Flask and other dependencies
RUN pip install flask mlflow

# Set the environment variable for the main Flask app
ENV FLASK_APP=app.py

# Run the application when the container starts
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
