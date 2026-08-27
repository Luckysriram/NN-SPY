.PHONY: setup test feasibility audit clean

setup:
	python -m venv .venv
	.venv/Scripts/python -m pip install --upgrade pip
	.venv/Scripts/python -m pip install -e ".[dev]"

test:
	python -m pytest -q

feasibility:
	python -m data.adapters.feasibility_check $(SOURCE)

audit:
	python -m labels.audit --n 50 --seed 42

clean:
	rm -rf .pytest_cache **/__pycache__
