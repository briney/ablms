# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run a specific test file
pytest tests/test_config.py

# Run a specific test class or method
pytest tests/test_config.py::TestModelRegistry
pytest tests/test_config.py::TestModelRegistry::test_models_registered

# Run tests with verbose output
pytest -v

# Linting and formatting
black src/ tests/
ruff check src/ tests/
mypy src/
```

## Code Style

- **Type hints**: Use Python 3.10+ union syntax. Write `x | None` instead of `Optional[x]` and `x | y` instead of `Union[x, y]`.
- **Imports**: Do not import `Optional` or `Union` from `typing`.

## Architecture

ablms provides a unified Python API for multiple antibody language models with different architectures.

### Class Hierarchy

```
BaseAbLM (abstract base)
├── EncoderAbLM (encoder models)
│   ├── IgBERT
│   ├── IgT5 (no MLM head)
│   ├── AntiBERTa2
│   ├── BALM
│   ├── AntiBERTy (single-chain only)
│   └── AbLang2
└── GenerativeAbLM (generative models)
    └── IgLM
```

### Key Components

- **`AntibodySequence`** (`core/sequence.py`): Unified sequence representation with `<MASK>` token that gets converted to model-specific tokens. All model methods accept sequences in this format or as raw strings.

- **`ModelConfig` and registry** (`core/config.py`): Models are registered in `MODEL_REGISTRY` with configuration (supports_paired, max_length, mask_token, etc.). Use `load_model("name")` to instantiate.

- **Multi-GPU Execution** (`parallel/`): `MultiGPUExecutor` manages worker processes for parallel inference. Single-GPU mode skips subprocess overhead. Workers are lazily initialized on first inference call.

### Adding a New Encoder Model

1. Create `src/ablms/encoders/yourmodel.py` inheriting from `EncoderAbLM`
2. Set class attributes: `model_name`, `supports_paired`, `max_length`, `embedding_dim`, `mask_token`, `separator`, `has_mlm_head`
3. Implement abstract methods:
   - `_load_model()`: Load model and tokenizer
   - `_format_for_model()`: Convert `<MASK>` to model-specific token, join chains
   - `_tokenize()`: Tokenize formatted strings
   - `_forward_embeddings()`, `_forward_all_hidden_states()`, `_forward_attention()`, `_forward_logits()`: Forward passes
   - `_get_vocab()`, `_compute_pseudo_ll()`, `_fill_mask_batch()`: MLM-related methods
4. Register in `core/config.py::_register_all_models()`
5. Export from `encoders/__init__.py` and `__init__.py`

### Important Patterns

- **Unified mask token**: All models use `<MASK>` internally. Each model's `_format_for_model()` converts to its native token (`[MASK]`, `_`, `<mask>`, `*`).
- **Input normalization**: `_normalize_input()` in `BaseAbLM` accepts strings, `AntibodySequence`, or lists of either.
- **Token offsets**: `_compute_token_offsets()` returns chain positions for extracting chain-specific embeddings from output tensors.
- **Batch processing methods**: Public methods call executor; `_process_*_batch()` methods are called by workers and should not parallelize further.

### Output Classes

Located in `outputs/`:
- `EmbeddingOutput`: Token/sequence embeddings with `get_chain_embeddings()`
- `LogitsOutput`: MLM logits with `probabilities`, `predictions`, `top_k_predictions()`
- `AttentionOutput`: Attention weights with `get_layer()`, `get_head()`, `get_mean_attention()`
- `GenerationOutput`: Generated sequences with `get_top_k()`, `filter_by_score()`

### Model-Specific Notes

- **IgT5**: `has_mlm_head = False` - raises `UnsupportedOperationError` for `get_logits()`, `fill_mask()`, `pseudo_log_likelihood()`
- **AntiBERTy**: `supports_paired = False` - raises `PairedSequenceError` for paired sequences
