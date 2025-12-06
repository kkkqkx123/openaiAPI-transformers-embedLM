# AGENTS.md

## Build/Test Commands

This project uses uv + venv to manage dependecies. So when you want to use python command, always use `uv run <python command>` instead simple `<python command>`.

```bash
# Install all dependencies
uv sync

# Run all tests
uv run pytest

# Install dependencies
uv add <package-name>

# Run a single test
uv run pytest tests/test_e2e.py::TestE2EAPIFlow::test_complete_embedding_flow -v

# Run test file
uv run pytest tests/test_e2e.py -v

# Start dev server
uv run python -m emb_model_provider.main
```

## Architecture & Structure

- **Framework**: FastAPI with asyncio/uvicorn
- **ML**: PyTorch + transformers
- **Config**: Pydantic BaseSettings with env vars (EMB_PROVIDER_* prefix)
- **Main package**: `emb_model_provider/`
  - `main.py`: FastAPI app entry point
  - `api/`: Route handlers (embeddings, models, middleware, exceptions)
  - `core/`: Config, logging, model_manager, tokenizer_manager, performance_monitor
  - `services/`: EmbeddingService, batch optimization logic
- **Tests**: `tests/` - e2e, unit, performance, compatibility tests
- **API**: OpenAI-compatible `/v1/embeddings` and `/v1/models` endpoints

## Code Style & Conventions

- **Python version**: 3.10+
- **Type hints**: mypy enforced (`disallow_untyped_defs=true`)
- **Testing**: pytest with asyncio support
- **Error handling**: Custom OpenAI-compatible error responses in `api/exceptions.py`
- **Logging**: JSON structured logging via `core/logging.py`
- **Config**: All user config via env vars. Check `.env` file for all available variables.
