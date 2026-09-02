PY := uv run python

.PHONY: install data sim report figures benchmark test lint clean

install:
	uv sync

data:
	$(PY) -m abtest_sim.cli data

sim:
	$(PY) -m abtest_sim.cli sim

figures:
	$(PY) -m abtest_sim.cli figures

benchmark:
	$(PY) -m abtest_sim.cli benchmark

report: figures
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/report.ipynb

test:
	uv run pytest

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

clean:
	rm -f results/*.parquet figures/*.png
