# Created with the help of ChatGPT
# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN apt-get update && apt-get install -y build-essential
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Create a directory for the cache outside the container and mount it
VOLUME /app/cache

# Expose the port where Streamlit will run the app
EXPOSE 8501

# Define the command to run your Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port", "8501"]
