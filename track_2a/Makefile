.PHONY: run eval synthetic train

# legacy = nvidia runtime available, cdi = only CDI devices available
GPU_MODE ?= $(shell \
  if docker info 2>/dev/null | grep -qi 'Runtimes:.*nvidia'; then echo legacy; \
  elif docker info 2>/dev/null | grep -q 'nvidia.com/gpu=all'; then echo cdi; \
  else echo legacy; fi)

ifeq ($(GPU_MODE),cdi)
COMPOSE := docker compose -f compose.yaml -f compose.cdi.yaml
else
COMPOSE := docker compose -f compose.yaml
endif

run: eval

eval: .env
	docker compose build eval
	docker compose run --rm eval

synthetic: .env
	$(COMPOSE) --profile gpu build synthetic
	$(COMPOSE) --profile gpu run --rm synthetic

train: .env
	@test -f src/synthetic_data/synthetic/train_mixed.jsonl || { echo "No training data found: run 'make synthetic' first"; exit 1; }
	$(COMPOSE) --profile gpu build train
	$(COMPOSE) --profile gpu run --rm train

.env:
	cp .env.example .env
	@echo "Created .env from .env.example - edit it to change model, endpoint or split"
