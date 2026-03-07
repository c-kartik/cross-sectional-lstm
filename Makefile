PYTHON ?= python3

.PHONY: test run-walkforward run-opt-rd run-opt-grid data-validate

test:
	$(PYTHON) -m pytest -q

run-walkforward:
	$(PYTHON) -m src.experiments.run_walk_forward

run-opt-rd:
	$(PYTHON) -m src.experiments.run_lstm_optimize_portfolio

run-opt-grid:
	$(PYTHON) -m src.experiments.run_lstm_optimize_grid

data-validate:
	$(PYTHON) -m src.data.validate_prices
