# Use an official Python runtime as a parent image
FROM python:3.10-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Install system-level dependencies required for building PyAudio and other potential packages
# 'build-essential' provides tools like gcc, make, etc.
# 'libsndfile1' is often needed for audio processing libraries
# 'portaudio19-dev' is specifically for PyAudio
# Clean up apt caches to keep image size small
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    portaudio19-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Expose the port that the application will run on
EXPOSE 7860

# Define environment variables if needed by the application
# ENV SOME_VARIABLE="some_value"

# Run the command to start the application
CMD ["RUN_APP.sh"]
