@echo off
setlocal enabledelayedexpansion

:: Check if help is requested
if "%1"=="/?" goto :help
if "%1"=="-h" goto :help
if "%1"=="--help" goto :help
if "%1"=="" goto :help

:: Set environment variables
set "PROJECT_NAME=nlp_project"
set "CONTAINER_NAME=nlp_container"
set "PORT=8080"

:: Process commands
goto :%1 2>nul
echo Unknown command: %1
goto :help

:help
echo NLP Project Build Script
echo.
echo Usage: build.bat [command]
echo.
echo Available commands:
echo   setup             Set up project environment and install dependencies
echo   clean             Remove virtual environment and cached files
echo   build             Build the Docker image
echo   run               Run the container
echo   stop              Stop and remove the container
echo   docker-logs       Show container logs
echo   restart           Restart the container
echo   ps                Show running containers
echo   exec              Open a shell inside the container
echo   test              Run tests
echo   lint              Run linting
echo   show-stats-preprocessing  Show performance stats of preprocessing class
echo   run-preprocessing         Run preprocessing class functions
echo   run-feature_eng           Run feature engineering class functions
goto :eof

:show-stats-preprocessing
echo Showing performance stats of preprocessing class...
tuna %cd%\stats\results.prof
goto :eof

:run-preprocessing
echo Running preprocessing class functions...
if exist data\PREPROCESSED_Reviews.csv del data\PREPROCESSED_Reviews.csv
python -m src.preprocessing_model
goto :eof

:run-feature_eng
echo Running feature engineering class functions...
if exist data\bag_of_words.csv del data\bag_of_words.csv
python -m src.feature_engineering
goto :eof

:setup
echo Setting up project environment...
call %0 clean
python -m venv venv || (
    echo Failed to create virtual environment
    exit /b 1
)
call .\venv\Scripts\activate || (
    echo Failed to activate virtual environment
    exit /b 1
)
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m nltk.downloader punkt
python -m nltk.downloader punkt_tab
if not exist logs mkdir logs
if not exist data mkdir data
if not exist processed_data mkdir processed_data
echo Setup complete!
goto :eof

:clean
echo Cleaning project...
if exist venv rmdir /s /q venv
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc 2>nul
del /s /q .DS_Store 2>nul
echo Clean complete!
goto :eof

:build
echo Building Docker image...
docker build -t %PROJECT_NAME% . || (
    echo Failed to build Docker image
    exit /b 1
)
echo Build complete!
goto :eof

:run
echo Starting container...
docker start %CONTAINER_NAME% 2>nul || (
    docker run -d --name %CONTAINER_NAME% ^
        -p %PORT%:%PORT% ^
        -v %cd%\data:/app/data ^
        -v %cd%\processed_data:/app/processed_data ^
        -v %cd%\logs:/app/logs ^
        %PROJECT_NAME%
)
echo Container is running on port %PORT%
goto :eof

:stop
echo Stopping container...
docker stop %CONTAINER_NAME% 2>nul
docker rm %CONTAINER_NAME% 2>nul
echo Container stopped and removed
goto :eof

:docker-logs
echo Showing container logs...
docker logs -f %CONTAINER_NAME%
goto :eof

:restart
call %0 stop
call %0 run
goto :eof

:ps
echo Showing running containers...
docker ps -a
goto :eof

:exec
echo Opening shell inside the container...
docker exec -it %CONTAINER_NAME% /bin/bash
goto :eof

:test
echo Running tests...
call .\venv\Scripts\activate
python -m pytest tests/
goto :eof

:lint
echo Running linting...
call .\venv\Scripts\activate
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
goto :eof

:eof
endlocal