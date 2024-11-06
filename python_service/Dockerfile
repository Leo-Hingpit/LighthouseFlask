FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements.txt
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose the port your app will run on
EXPOSE 8080

# Use Gunicorn to run the app
CMD ["python", "app.py"]

#CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]
