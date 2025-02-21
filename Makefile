.PHONY: help build run stop logs restart ps exec

help:
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?#"} /^[a-zA-Z_-]+:.*?#/ {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build:  # Build the Docker image
	docker build -t nlp_project .

run:  # Run the container
	@docker start nlp_container 2>/dev/null || docker run -d --name nlp_container -p 8080:8080 nlp_project

stop:  # Stop and remove the container
	docker stop nlp_container && docker rm nlp_container

docker-logs:  # Show container logs
	docker logs -f nlp_container

restart:  # Restart the container
	make stop && make run

ps:  # Show running containers
	docker ps -a

exec:  # Open a shell inside the container
	docker exec -it nlp_container /bin/sh
