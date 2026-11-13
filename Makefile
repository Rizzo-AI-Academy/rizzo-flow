# Rizzo Flow API in a container — docker or podman, CPU or GPU.
#
# The image is built from this checkout (no commit to pin); the llama.cpp build and the GGUF
# are downloaded into ./data on first start, so `make up` is the only slow first step.
#
# Every target honours COMPOSE="<compose command>"; by default it uses `docker compose` when
# docker is installed and `podman compose` otherwise.

COMPOSE ?= $(shell if command -v docker >/dev/null 2>&1; then echo docker compose; else echo podman compose; fi)
GPU := -f docker-compose.yml -f docker-compose.gpu.yml
CUDA := -f docker-compose.yml -f docker-compose.cuda.yml
PORT ?= 8017
IMAGE ?= rizzo-flow:local

.DEFAULT_GOAL := help
.PHONY: help build up gpu cuda down stop restart logs ps shell health update clean

help:  ## This list
	@grep -hE '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sort | awk -F':.*## ' '{printf "  make %-8s %s\n", $$1, $$2}'

build:  ## Build the image from this checkout
	$(COMPOSE) build

up: build  ## Build and start the API on 127.0.0.1:8017 (CPU)
	$(COMPOSE) up -d
	@echo "rizzo: http://127.0.0.1:$(PORT)/playground — the first start downloads the runtime and the weights (~1.2 GB with RIZZO_SIZE=1.7b RIZZO_QUANT=q4_k_m, ~4.4 GB with the defaults); follow it with: make logs"

gpu:  ## Build and start with the host GPU through Vulkan (AMD/Intel)
	$(COMPOSE) $(GPU) build
	$(COMPOSE) $(GPU) up -d

cuda:  ## Build and start with an NVIDIA GPU through CUDA
	$(COMPOSE) $(CUDA) build
	$(COMPOSE) $(CUDA) up -d

down:  ## Stop and remove the container (the ./data volume survives)
	$(COMPOSE) down

stop:  ## Stop the container, keep it
	$(COMPOSE) stop

restart:  ## Restart the container
	$(COMPOSE) restart

logs:  ## Follow the container log
	$(COMPOSE) logs -f --tail=100

ps:  ## Show the container status and health
	$(COMPOSE) ps

shell:  ## A shell inside the image (models and runtime mounted)
	$(COMPOSE) run --rm rizzo sh

health:  ## GET /health on the running container
	@curl -fsS -m 5 http://127.0.0.1:$(PORT)/health && echo

update:  ## Rebuild the image from the current checkout and restart
	$(COMPOSE) build --pull
	$(COMPOSE) up -d
	$(COMPOSE) ps

clean: down  ## Remove the container, the image and ./data (weights and runtime)
	$(COMPOSE) rm -f
	-$(firstword $(COMPOSE)) image rm $(IMAGE)
	rm -rf data
