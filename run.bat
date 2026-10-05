@echo off
setlocal

set BASE=https://raw.githubusercontent.com/Minuicee/Project_20/main

:: -------------------------------
:: Activate virtual environment
:: -------------------------------
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Please run install.bat first.
    pause
    exit /b
)
call venv\Scripts\activate

:: -------------------------------
:: Install requirements if missing
:: -------------------------------
if not exist "requirements\requirements.txt" (
    echo requirements.txt not found. Downloading...
    if not exist "requirements" mkdir requirements
    powershell -Command "Invoke-WebRequest -Uri %BASE%/requirements/requirements.txt -OutFile requirements\requirements.txt"
    echo Installing required packages...
    pip install --upgrade pip
    pip install -r requirements\requirements.txt
)

:: -------------------------------
:: Download main.py if missing
:: -------------------------------
if not exist "main.py" (
    echo main.py not found. Downloading...
    powershell -Command "Invoke-WebRequest -Uri %BASE%/main.py -OutFile main.py"
)

:: -------------------------------
:: Download new datasets
:: -------------------------------
echo Checking for new datasets...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $base='%BASE%'; $sets=Invoke-RestMethod -Uri 'https://api.github.com/repos/Minuicee/Project_20/contents/sets?ref=main'; foreach ($set in $sets | Where-Object { $_.type -eq 'dir' }) { $path=Join-Path 'sets' $set.name; if (-not (Test-Path $path)) { New-Item -ItemType Directory -Force -Path $path | Out-Null; Invoke-WebRequest -Uri ($base + '/sets/' + $set.name + '/language1.csv') -OutFile (Join-Path $path 'language1.csv'); Invoke-WebRequest -Uri ($base + '/sets/' + $set.name + '/language2.csv') -OutFile (Join-Path $path 'language2.csv'); Write-Host ('Downloaded new set: ' + $set.name) } }"
if errorlevel 1 echo Could not check for new datasets. Continuing with installed datasets.

:: -------------------------------
:: Run without terminal window
:: -------------------------------
if exist "venv\Scripts\pythonw.exe" (
    start "" "venv\Scripts\pythonw.exe" main.py
) else (
    echo pythonw.exe not found in venv.
    pause
)