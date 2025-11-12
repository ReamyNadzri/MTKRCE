@echo off
REM Navigate to the directory
cd MTKRCE

REM Set the environment variable
set GEMINI_API_KEY=AIzaSyAMMdP5Jl-fJ91cDu9rM4A3jKBF3tp5awc
set MONGO_DB_PASSWORD=xDW9wopR0U8oFQFH

REM Run the Python application
python app.py

REM Pause the window so it doesn't close immediately after finishing (helpful for seeing errors)
pause