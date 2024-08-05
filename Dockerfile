FROM python:3.10-slim

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libssl-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# set working directory
WORKDIR /app

# copy current directory -> /app container
COPY . /app

# packages from requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# external port
EXPOSE 5000

#  environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# run the application
CMD ["flask", "run"]