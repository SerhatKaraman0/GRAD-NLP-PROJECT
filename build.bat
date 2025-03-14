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
echo   setup      Set up project environment and install dependencies
echo   clean      Remove virtual environment and cached files
echo   build      Build the Docker image
echo   run        Run the container
echo   stop       Stop and remove the container
echo   logs       Show container logs
echo   restart    Restart the container
echo   ps         Show running containers
echo   exec       Open a shell inside the container
echo   test       Run tests
echo   lint       Run linting
goto :eof

:setup
echo Setting up project environment...
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
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt
python -m nltk.downloader punkt_tab
mkdir logs 2>nul
mkdir data 2>nul
mkdir processed_data 2>nul
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

:logs
docker logs -f %CONTAINER_NAME%
goto :eof

:restart
call %0 stop
call %0 run
goto :eof

:ps
docker ps -a --filter "name=%CONTAINER_NAME%"
goto :eof

:exec
docker exec -it %CONTAINER_NAME% /bin/bash
goto :eof

:test
echo Running tests...
call .\venv\Scripts\activate
python -m pytest tests/ -v
goto :eof

:lint
echo Running linting...
call .\venv\Scripts\activate
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
goto :eof

:eof
endlocal