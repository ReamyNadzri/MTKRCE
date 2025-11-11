@echo off
REM Navigate to the directory
cd MTKRCE

REM Set the environment variable
set GEMINI_API_KEY=AIzaSyA2gNgczTmAhdJtRl4CnjSQQ6Dp5TFAqg8

REM Run the Python application
python app.py

REM Pause the window so it doesn't close immediately after finishing (helpful for seeing errors)
pause