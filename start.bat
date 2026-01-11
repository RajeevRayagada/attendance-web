@echo off
call .venv\Scripts\activate
start cmd /k python app.py
timeout /t 3
start cmd /k ngrok http 5000
