@echo off
REM Activate the virtual environment and start the Image Analysis UI
cd /d %~dp0
call .venv\Scripts\activate.bat
python -m src.ui.main_window_v2
