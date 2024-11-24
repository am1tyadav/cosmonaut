help:
	@echo "Available commands:"
	@echo "  install      	- Install the package in active environment"
	@echo "  single_label 	- Run single label example"
	@echo "  multi_label  	- Run multi label example"
	@echo "  multi_category - Run multi category example"
	@echo "  examples       - Run examples"
	@echo "  lint           - Run trunk"

install:
	python -m pip install ".[all]"

single_label:
	python examples/single_label/main.py

multi_label:
	python examples/multi_label/main.py

multi_category:
	python examples/multi_category/main.py

examples: single_label multi_label multi_category

lint:
	trunk check --fix --all

.DEFAULT_GOAL := help
