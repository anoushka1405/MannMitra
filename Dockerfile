# Use a lightweight base image with Python
FROM python:3.10-slim

# Avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Create a working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project
COPY . .

# Expose the port your app runs on
EXPOSE 8080

# Run the Flask app using Gunicorn (faster than `flask run`)
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
