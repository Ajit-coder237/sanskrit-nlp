.PHONY: install test lint typecheck cli

VENV := .venv/bin/

install:
	$(VENV)python -m pip install -e ".[dev]"

test:
	$(VENV)python -m pytest -q

lint:
	$(VENV)ruff check src tests

typecheck:
	$(VENV)mypy src/sanskrit_nlp --ignore-missing-imports

cli:
	$(VENV)python -m sanskrit_nlp.cli --help
