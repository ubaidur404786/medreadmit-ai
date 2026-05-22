COMPOSE = docker compose -f docker/docker-compose.yml

.PHONY: up down logs rebuild

up:
	$(COMPOSE) up --build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

rebuild:
	$(COMPOSE) build --no-cache
