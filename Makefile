.PHONY: help show-stats-preprocessing run-preprocessing setup clean build run stop logs restart ps exec test lint

help:  # Show this help message
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?#"} /^[a-zA-Z_-]+:.*?#/ {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

show-stats-preprocessing: # shows performance stats of preprocessing class
	tuna $(PWD)/stats/results.prof
	
run-preprocessing: # run preprocessing class functions
	rm -f PREPROCESSED_Reviews.csv
	python3 -m src.preprocessing_model

setup: clean  # Set up project environment and install dependencies
	python3 -m venv venv
	. venv/bin/activate && \
	pip3 install --upgrade pip && \
	pip3 install -r requirements.txt && \
	python3 -m nltk.downloader punkt && \
	python3 -m nltk.downloader punkt_tab && \
	mkdir -p logs data processed_data

clean:  # Remove virtual environment and cached files
	rm -rf venv
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete

build:  # Build the Docker image
	docker build -t nlp_project .

run:  # Run the container
	@docker start nlp_container 2>/dev/null || docker run -d \
		--name nlp_container \
		-p 8080:8080 \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/processed_data:/app/processed_data \
		-v $(PWD)/logs:/app/logs \
		nlp_project

stop:  # Stop and remove the container
	docker stop nlp_container && docker rm nlp_container

docker-logs:  # Show container logs
	docker logs -f nlp_container

restart:  # Restart the container
	make stop && make run

ps:  # Show running containers
	docker ps -a

exec:  # Open a shell inside the container
	docker exec -it nlp_container /bin/bash

test:  # Run tests
	. venv/bin/activate && python -m pytest tests/

