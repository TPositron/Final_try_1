@echo off
echo Starting Image Analysis GUI...

REM Add the current directory to Python path
set PYTHONPATH=%CD%;%PYTHONPATH%

REM Run the main window module
python -m src.image_analysis.ui.main_window

if errorlevel 1 (
    echo.
    echo Error occurred. See details above.
    pause
)
