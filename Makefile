.PHONY: test gate assets lint docker
PYTHON ?= python

test:
	$(PYTHON) -m pytest tests -q

gate:
	$(PYTHON) tools/check_readme_numbers.py

assets:
	$(PYTHON) tools/verify_assets.py

lint:
	$(PYTHON) -m ruff check src scripts tools tests

docker:
	docker build -t mario64ds-rl .
