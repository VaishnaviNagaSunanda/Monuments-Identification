FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and building python packages
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Setup permissions for Hugging Face Spaces (runs as non-root user 1000)
RUN chmod +x start.sh
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

# Collect static files and set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

EXPOSE 7860

# Command to run on start
CMD ["/bin/bash", "./start.sh"]
