.PHONY: help setup download-data download-weights format lint evaluate clean

PYTHON := uv run python
TARGET_MODEL := google/gemma-3-4b-pt
DRAFT_MODEL := google/gemma-3-1b-pt

help:
	@echo "Available commands:"
	@echo "  make setup          - Install project dependencies via uv"
	@echo "  make download-data  - Download and cache the WikiText-2 dataset via HuggingFace"
	@echo "  make download-weights - Download and cache Gemma model weights via HuggingFace"
	@echo "  make format         - Auto-format code and fix linting errors"
	@echo "  make lint           - Check code for linting, formatting, and type issues"
	@echo "  make evaluate       - Run the Speculative Decoding evaluation pipeline"
	@echo "  make clean          - Remove cache, logs, and temporary files"

setup:
	uv sync --all-extras

download-data:
	@echo "Downloading WikiText-2 dataset..."
	$(PYTHON) -c "from datasets import load_dataset; load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1')"

download-weights:
	@echo "Downloading weights for $(TARGET_MODEL) and $(DRAFT_MODEL)..."
	$(PYTHON) -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
	AutoTokenizer.from_pretrained('$(TARGET_MODEL)'); \
	AutoModelForCausalLM.from_pretrained('$(TARGET_MODEL)'); \
	AutoModelForCausalLM.from_pretrained('$(DRAFT_MODEL)')"

format:
	uv run ruff check --fix .
	uv run ruff format .

lint:
	uv run ruff check .
	uv run ruff format --check .

evaluate:
	@echo "Running Speculative Decoding Pipeline..."
	$(PYTHON) -m src.evaluate --target_model $(TARGET_MODEL) --draft_model $(DRAFT_MODEL)

clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache wandb logs
	find . -type d -name "__pycache__" -exec rm -rf {} +