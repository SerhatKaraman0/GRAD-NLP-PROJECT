@echo off
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

if "%1"=="setup" (
    python -m venv venv
    call .\venv\Scripts\activate
    pip install --upgrade pip
    pip install -r requirements.txt
    python -m nltk.downloader punkt
    python -m nltk.downloader punkt_tab
    mkdir logs data processed_data
) else if "%1"=="clean" (
    rmdir /s /q venv
    rmdir /s /q __pycache__
    del /s /q *.pyc
    del /s /q .DS_Store
) else if "%1"=="build" (
    docker build -t nlp_project .
) else if "%1"=="run" (
    docker start nlp_container 2>nul || docker run -d --name nlp_container -p 8080:8080 -v %cd%\data:/app/data -v %cd%\processed_data:/app/processed_data -v %cd%\logs:/app/logs nlp_project
) else if "%1"=="stop" (
    docker stop nlp_container
    docker rm nlp_container
) else if "%1"=="logs" (
    docker logs -f nlp_container
) else if "%1"=="restart" (
    call build.bat stop
    call build.bat run
) else if "%1"=="ps" (
    docker ps -a
) else if "%1"=="exec" (
    docker exec -it nlp_container /bin/bash
) else if "%1"=="test" (
    call .\venv\Scripts\activate
    python -m pytest tests/
) else if "%1"=="lint" (
    call .\venv\Scripts\activate
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
) else (
    echo Unknown command: %1
)
