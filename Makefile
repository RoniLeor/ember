MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables
.DEFAULT_GOAL := help

.PHONY: help sync check check-lint format fix test test-verbose build

help:
	@echo "Targets: sync check check-lint format fix test test-verbose build"
	@echo 'Options: UV_SYNC_ARGS="--extra gpu"; PYTEST_ARGS="tests/playground/breakout/models.py -k invalid"'

UV_SYNC_ARGS ?=
PYTEST_ARGS ?=

sync:
	uv sync --locked --inexact $(UV_SYNC_ARGS)

check: check-lint test

check-lint:
	uv run --no-sync ruff format --check .
	uv run --no-sync ruff check .
	uv run --no-sync pyright
	uv run --no-sync ty check
	uv run --no-sync mypy

format:
	uv run --no-sync ruff format .

fix: format
	uv run --no-sync ruff check --fix .

test:
	uv run --no-sync pytest -q $(PYTEST_ARGS)

test-verbose:
	uv run --no-sync pytest -vv $(PYTEST_ARGS)

build:
	uv build --wheel
