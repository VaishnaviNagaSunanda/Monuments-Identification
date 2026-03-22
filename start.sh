#!/bin/bash
python manage.py collectstatic --noinput
python manage.py migrate
gunicorn Monuments_Identification.wsgi:application --bind 0.0.0.0:7860 --timeout 300 --workers 1
