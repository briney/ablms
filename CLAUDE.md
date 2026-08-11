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

# Reproduce CI's blocking test job (excludes tests that load real model weights)
python -m pytest -m "not slow"

# Reproduce CI's non-blocking smoke job (bundled-weight models, offline).
# Currently expected to fail: IgLM and AntiBERTy are broken under
# transformers 5.x (tracking issue #5).
python -m pytest -m smoke

# Linting and formatting
black src/ tests/
ruff check src/ tests/
ty check src/
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
2. Set class attributes: `model_name`, `supports_paired`, `max_length`, `embedding_dim`,
   `mask_token`, `separator`, `has_mlm_head`. Override the `num_layers` property if the
   model's config does not expose `num_hidden_layers` under that name, or has no
   HuggingFace config at all (IgT5 and AbLang2 both need this override), and set
   `supports_intermediate_layers = False` if only the final layer is reachable (AbLang).
3. Implement abstract methods:
   - `_load_model()`: Load model and tokenizer
   - `_format_for_model()`: Convert `<MASK>` to model-specific token, join chains
   - `_tokenize()`: Tokenize formatted strings
   - `_forward_embeddings()`, `_forward_all_hidden_states()`, `_forward_attention()`, `_forward_logits()`: Forward passes
   - `_get_vocab()`, `_compute_pseudo_ll()`, `_fill_mask_batch()`: MLM-related methods
4. Register in `core/config.py::_register_all_models()`
5. Export from `encoders/__init__.py` and `__init__.py`

**Overriding a `_process_*_batch` method:** if a model needs its own version of
one of these (AbLang does, because heavy and light chains go through separate
model heads), the override's signature must stay bind-compatible with the base
class's. `MultiGPUExecutor.execute()` forwards every extra argument through
`**method_kwargs` by keyword, so a base-class parameter missing from the
override becomes a `TypeError` at inference time rather than at import time.
`tests/test_encoder_contract.py` enforces this with `inspect.signature`.

### Important Patterns

- **Unified mask token**: All models use `<MASK>` internally. Each model's `_format_for_model()` converts to its native token (`[MASK]`, `_`, `<mask>`, `*`).
- **Input normalization**: `_normalize_input()` in `BaseAbLM` accepts strings, `AntibodySequence`, or lists of either.
- **Token offsets**: `_compute_token_offsets()` returns chain positions for extracting chain-specific embeddings from output tensors.
- **Batch processing methods**: Public methods call executor; `_process_*_batch()` methods are called by workers and should not parallelize further.
- **Reduce before transfer**: `_process_*_batch()` methods run in worker processes and
  return through a queue backed by `/dev/shm`. Any reduction that shrinks the result
  (pooling, scoring) belongs inside the batch method, before `.cpu()`, not after the
  executor concatenates. See `EncoderAbLM._process_embeddings_batch`.
- **Streaming variants**: `MultiGPUExecutor.execute_iter()` yields `(batch_index, result)`
  in input order with a bounded submission window; `execute()` is a thin wrapper that
  combines it. Public streaming APIs (e.g. `iter_embeddings()`) build on `execute_iter`.
- **Layer selection**: `get_embeddings(layer=...)` accepts an int, a list of ints, or
  `"all"`. `utils/layers.py::resolve_layer_selection` validates and resolves it in the
  parent before dispatch; a list return means the batch method stacks a layer axis at
  dimension 1. Multi-layer requests route through `_forward_all_hidden_states()`, which
  every encoder already implements, so no encoder needs a multi-layer forward pass. Pooling
  is applied per layer *before* stacking, so the token-level tensor is never built on
  pooled runs.

### Output Classes

Located in `outputs/`:
- `EmbeddingOutput`: Token/sequence embeddings with `get_chain_embeddings()`. Multi-layer
  results carry a layer axis at dimension 1, `layers` listing the resolved indices, and
  `get_layer()` / `concat_layers()` for extracting or flattening it.
- `LogitsOutput`: MLM logits with `probabilities`, `predictions`, `top_k_predictions()`
- `AttentionOutput`: Attention weights with `get_layer()`, `get_head()`, `get_mean_attention()`
- `GenerationOutput`: Generated sequences with `get_top_k()`, `filter_by_score()`

### Model-Specific Notes

- **IgT5**: `has_mlm_head = False` - raises `UnsupportedOperationError` for `get_logits()`, `fill_mask()`, `pseudo_log_likelihood()`
- **AntiBERTy**: `supports_paired = False` - raises `PairedSequenceError` for paired sequences
