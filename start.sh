#!/bin/bash
# Collect static files
python manage.py collectstatic --noinput

# Run database migrations
python manage.py migrate

# Start the gunicorn server on port 7860 (required by Hugging Face)
gunicorn Monuments_Identification.wsgi:application --bind 0.0.0.0:7860
