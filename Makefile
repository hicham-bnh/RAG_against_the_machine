PYTHON = python3.10
PROJECT_NAME = RAG
UV = uv
REQUIREMENTS = requirements

help:
	@echo "Available commands:"
	@echo " make setup                            - Creat virtual environment and install build tools"
	@echo " make install                          - Install project dependencies"
	@echo " make run <Optional FILE=filename>     - Execute the main script"
	@echo " make debug                            - Run the main script in debug mode (pdb)"
	@echo " make clean                            - Remove temporary files and caches"
	@echo " make fclean                           - Remove temporary files, caches and virtual environmnet"
	@echo " make lint                             - Run flake8 amd mypy with standard checks"
	@echo " make lint-strict                      - Rune flake8 and mypy with strict mode"
	@echo " make build                            - Built the python package"
	@echo " make help                             - show this help message"


install:
	$(UV) sync

run:
	$(UV) run python -m student

debug:
	$(PYTHON) -m pdb $(MAIN_SCRIPT) $(FILE)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name *.egg-info -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf build/ dist/ 2>/dev/null || true

fclean:
	@rm -rf $(VENV) 2>/dev/null || true

lint:
	$(PYTHON) -m flake8 . --exclude=.git,data/raw,.VENV,venv,.venv,env,test_vm,build,dist,.mypy_cache,.pytest_cache,__pycache__,dependencies,*.egg-info
	$(PYTHON) -m mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs --exclude='(data/raw|build|dist|venv|env|dependencies)'



.DEFAULT_GOAL := help

.PHONY: install run debug clean lint help