@echo off
setlocal

echo =============================
echo SRS AI Installer
echo =============================
echo.

:: Create and enter project folder
if not exist "spaced_repetition_system" mkdir spaced_repetition_system
cd spaced_repetition_system
if errorlevel 1 (echo Failed to enter project directory & pause & exit /b)

:: ---------------------------------
:: Create required directories
:: ---------------------------------
if not exist "data"  mkdir data
if not exist "img"   mkdir img
if not exist "sets"  mkdir sets

:: ---------------------------------
:: Download UI images
:: ---------------------------------
echo Downloading core files...

set BASE=https://raw.githubusercontent.com/Minuicee/Project_20/main

powershell -Command "Invoke-WebRequest -Uri %BASE%/img/folder_button.png   -OutFile img\folder_button.png   -ErrorAction SilentlyContinue"
powershell -Command "Invoke-WebRequest -Uri %BASE%/img/settings_button.png -OutFile img\settings_button.png -ErrorAction SilentlyContinue"
powershell -Command "Invoke-WebRequest -Uri %BASE%/img/edit_button.png     -OutFile img\edit_button.png     -ErrorAction SilentlyContinue"
powershell -Command "Invoke-WebRequest -Uri %BASE%/img/loop_button.png     -OutFile img\loop_button.png     -ErrorAction SilentlyContinue"
powershell -Command "Invoke-WebRequest -Uri %BASE%/img/gauss_button.png     -OutFile img\gauss_button.png     -ErrorAction SilentlyContinue"
powershell -Command "Invoke-WebRequest -Uri %BASE%/img/pause_button.png     -OutFile img\pause_button.png     -ErrorAction SilentlyContinue"
powershell -Command "Invoke-WebRequest -Uri %BASE%/img/stop_button.png     -OutFile img\stop_button.png     -ErrorAction SilentlyContinue"
powershell -Command "Invoke-WebRequest -Uri %BASE%/img/start_button.png     -OutFile img\start_button.png     -ErrorAction SilentlyContinue"
powershell -Command "Invoke-WebRequest -Uri %BASE%/img/folder_button.png     -OutFile img\folder_button.png     -ErrorAction SilentlyContinue"
powershell -Command "Invoke-WebRequest -Uri %BASE%/img/help_button.png     -OutFile img\help_button.png     -ErrorAction SilentlyContinue"
:: ---------------------------------
:: Download all available sets
:: ---------------------------------
echo.
echo Downloading available datasets...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $base='%BASE%'; $sets=Invoke-RestMethod -Uri 'https://api.github.com/repos/Minuicee/Project_20/contents/sets?ref=main'; foreach ($set in $sets | Where-Object { $_.type -eq 'dir' }) { $path=Join-Path 'sets' $set.name; New-Item -ItemType Directory -Force -Path $path | Out-Null; Invoke-WebRequest -Uri ($base + '/sets/' + $set.name + '/language1.csv') -OutFile (Join-Path $path 'language1.csv'); Invoke-WebRequest -Uri ($base + '/sets/' + $set.name + '/language2.csv') -OutFile (Join-Path $path 'language2.csv'); Write-Host ('Installed set: ' + $set.name) }"
if errorlevel 1 (echo Failed to download datasets. & pause & exit /b)

:: ---------------------------------
:: Check Python installation
:: ---------------------------------
python --version >nul 2>&1
if errorlevel 1 (echo Python is not installed. & pause & exit /b)

:: ---------------------------------
:: Create virtual environment
:: ---------------------------------
if not exist "venv" python -m venv venv
call venv\Scripts\activate

:: ---------------------------------
:: Install requirements
:: ---------------------------------
if not exist "requirements\requirements.txt" (
    echo requirements.txt not found. Downloading...
    if not exist "requirements" mkdir requirements
    powershell -Command "Invoke-WebRequest -Uri %BASE%/requirements/requirements.txt -OutFile requirements\requirements.txt"
    echo Installing packages...
    pip install --upgrade pip
    pip install -r requirements\requirements.txt
)

:: ---------------------------------
:: Update main.py if changed
:: ---------------------------------
powershell -Command "Invoke-WebRequest -Uri %BASE%/main.py -OutFile temp_main.py"

fc temp_main.py main.py >nul 2>&1
if errorlevel 1 (
    echo New version detected. Updating main.py...
    move /Y temp_main.py main.py
) else (
    del temp_main.py
)

:: ---------------------------------
:: Download run.bat if missing
:: ---------------------------------
if not exist "run.bat" (
    echo Downloading run.bat...
    powershell -Command "Invoke-WebRequest -Uri %BASE%/run.bat -OutFile run.bat"
) else (
    echo run.bat already exists.
)

echo.
echo Done.
pause