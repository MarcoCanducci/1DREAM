@echo off
REM 1DREAM GUI Launcher for Windows
REM This script launches the 1DREAM Toolbox GUI

echo =========================================
echo   1DREAM Toolbox - Manifold Learning
echo =========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

REM Change to the script directory
cd /d "%~dp0"

REM Launch the GUI
echo Starting 1DREAM GUI...
python 1DREAM_GUI.py

if errorlevel 1 (
    echo.
    echo An error occurred. Check that all dependencies are installed:
    echo   pip install numpy pandas scipy scikit-learn networkx matplotlib
    echo.
    pause
)
