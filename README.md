# ablms

A unified Python API for antibody language models.

## Overview

Working with antibody language models often means dealing with different architectures, tokenizers, input formats, and output structures. **ablms** provides a consistent interface across multiple models, so you can focus on your research instead of wrestling with model-specific quirks.

```python
from ablms import AntibodySequence, load_model

# Same API for any model
model = load_model("igbert")  # or "antiberty", "ablang2", "iglm", etc.
embeddings = model.get_embeddings(["EVQLVESGGGLVQPGRSLRL..."])
```

### Supported Models

| Model | Type | Paired Sequences | Source |
|-------|------|------------------|--------|
| **IgBERT** | Encoder | Yes | HuggingFace |
| **IgT5** | Encoder | Yes | HuggingFace |
| **AntiBERTa2** | Encoder | Yes | HuggingFace |
| **BALM** | Encoder | Yes | HuggingFace |
| **AntiBERTy** | Encoder | No | antiberty package |
| **AbLang2** | Encoder | Yes | ablang2 package |
| **ft-ESM** | Encoder | Yes | HuggingFace |
| **IgLM** | Generative | No | iglm package |

## Installation

```bash
pip install ablms
```

This installs ablms along with all required dependencies including PyTorch, Transformers, and the model-specific packages (antiberty, ablang2, iglm).

### From Source

```bash
git clone https://github.com/bryanbriney/ablms.git
cd ablms
pip install -e .
```

## Quickstart

### Creating Antibody Sequences

The `AntibodySequence` class provides a unified way to represent antibody sequences. All arguments must be passed as keywords to ensure the chain type is always explicit:

```python
from ablms import AntibodySequence, Species

# Single heavy chain
heavy_seq = AntibodySequence(heavy="EVQLVESGGGLVQPGRSLRLSCAASGFTFS")

# Single light chain
light_seq = AntibodySequence(light="DIQMTQSPSSLSASVGDRVTITCRASQSIS")

# Paired heavy and light chains
paired_seq = AntibodySequence(
    heavy="EVQLVESGGGLVQPGRSLRLSCAASGFTFS",
    light="DIQMTQSPSSLSASVGDRVTITCRASQSIS",
    species=Species.HUMAN
)

# Check sequence properties
print(paired_seq.is_paired)      # True
print(paired_seq.length)         # {'heavy': 30, 'light': 30}
print(paired_seq.total_length)   # 60
```

### Getting Embeddings

Extract residue-level or sequence-level embeddings from any encoder model:

```python
from ablms import load_model, AntibodySequence

# Load a model
model = load_model("igbert")

# Prepare sequences
sequences = [
    AntibodySequence(heavy="EVQLVESGGGLVQPGRSLRLSCAASGFTFS"),
    AntibodySequence(heavy="QVQLVQSGAEVKKPGASVKVSCKASGYTFT"),
]

# Get residue-level embeddings
output = model.get_embeddings(sequences)
print(output.embeddings.shape)  # [2, seq_len, 768]

# Get sequence-level embeddings with pooling
pooled = model.get_sequence_embeddings(sequences, pooling="mean")
print(pooled.embeddings.shape)  # [2, 768]
```

### Working with Paired Sequences

Models that support paired sequences (IgBERT, IgT5, BALM, AbLang2) can process heavy and light chains together:

```python
from ablms import load_model, AntibodySequence

model = load_model("igbert")  # Supports paired sequences

paired = AntibodySequence(
    heavy="EVQLVESGGGLVQPGRSLRLSCAASGFTFS",
    light="DIQMTQSPSSLSASVGDRVTITCRASQSIS"
)

output = model.get_embeddings([paired])

# Extract chain-specific embeddings
heavy_emb = output.get_chain_embeddings(0, "heavy")
light_emb = output.get_chain_embeddings(0, "light")
```

### Attention Weights

Visualize or analyze attention patterns:

```python
from ablms import load_model

model = load_model("igbert")
sequences = ["EVQLVESGGGLVQPGRSLRLSCAASGFTFS"]

attention = model.get_attention(sequences)
print(attention.num_layers)  # 12
print(attention.num_heads)   # 12

# Get attention from a specific layer and head
layer_5_head_0 = attention.get_head(layer=5, head=0)

# Get mean attention across all layers and heads
mean_attention = attention.get_mean_attention()
```

### Mask Filling

Predict amino acids at masked positions:

```python
from ablms import load_model, AntibodySequence

model = load_model("igbert")

# Create a sequence with masks
masked_seq = AntibodySequence(heavy="EVQL<MASK>ESGGGLVQPGRSLRL")

# Fill the mask with top predictions
predictions = model.fill_mask([masked_seq], top_k=5)

for pred in predictions[0]:
    print(pred.heavy_chain)
```

### Generating New Sequences

Use generative models like IgLM to create new antibody sequences:

```python
from ablms import load_model, ChainType, Species

model = load_model("iglm")

# Generate new heavy chain sequences
output = model.generate(
    num_sequences=5,
    chain_type=ChainType.HEAVY,
    species=Species.HUMAN,
    temperature=1.0
)

for seq in output.sequences:
    print(seq.heavy_chain)

# Get the best sequences by score
top_sequences = output.get_top_k(k=3)
```

### Computing Sequence Likelihoods

Score sequences using pseudo log-likelihood (encoder models) or log-likelihood (generative models):

```python
from ablms import load_model, AntibodySequence, ChainType, Species

# Encoder model: pseudo log-likelihood
encoder = load_model("igbert")
sequences = [
    AntibodySequence(heavy="EVQLVESGGGLVQPGRSLRL"),
    AntibodySequence(heavy="QVQLVQSGAEVKKPGASVKV"),
]
pll_scores = encoder.pseudo_log_likelihood(sequences)

# Generative model: log-likelihood
generator = load_model("iglm")
ll_scores = generator.log_likelihood(
    sequences,
    chain_type=ChainType.HEAVY,
    species=Species.HUMAN
)
```

## Key Concepts

### Unified Mask Token

All models use `<MASK>` as the mask token internally. ablms automatically converts this to each model's native mask token:

```python
# You always use <MASK>
seq = AntibodySequence(heavy="EVQL<MASK>ESGG")

# ablms converts it to the model's token:
# IgBERT: [MASK]
# AntiBERTy: _
# BALM: <mask>
# AbLang2: *
# ft-ESM: <mask>
```

### Output Classes

All methods return structured output objects with helpful properties:

- **`EmbeddingOutput`**: Token or sequence embeddings with `get_chain_embeddings()` for extracting specific chains
- **`LogitsOutput`**: MLM logits with `probabilities`, `predictions`, and `top_k_predictions()`
- **`AttentionOutput`**: Attention weights with `get_layer()`, `get_head()`, and `get_mean_attention()`
- **`GenerationOutput`**: Generated sequences with `get_top_k()` and `filter_by_score()`

### Device Management

Models automatically use all available GPUs for parallel inference:

```python
from ablms import load_model

# Auto-detects and uses all available GPUs
model = load_model("igbert")
print(model.num_devices)  # e.g., 4
print(model.devices)      # [device(type='cuda', index=0), ...]

# Or specify specific GPUs
model = load_model("igbert", devices=[0, 2, 3])

# Single GPU (no parallelization overhead)
model = load_model("igbert", devices="cuda:0")

# CPU only
model = load_model("igbert", devices="cpu")

# Move model after loading (resets to single device)
model.to("cuda:1")
```

### Multi-GPU Parallelism

When multiple GPUs are available, inference is automatically parallelized. Work is distributed across GPUs using a worker pool, with each GPU holding a complete model replica:

```python
from ablms import load_model, AntibodySequence

# Load model (auto-detects 4 GPUs)
model = load_model("igbert")

# Process 10,000 sequences - automatically distributed across GPUs
sequences = [AntibodySequence(heavy=seq) for seq in heavy_chains]
embeddings = model.get_embeddings(
    sequences,
    batch_size=64,       # Per-GPU batch size
    show_progress=True,  # tqdm progress bar (default: True)
)
```

Key features:
- **Automatic detection**: Uses all available GPUs by default
- **Lazy initialization**: Worker processes spawn on first inference call
- **Single-GPU optimization**: No subprocess overhead when using one device
- **Progress tracking**: Built-in tqdm progress bar for all inference methods

Disable the progress bar for cleaner output in scripts:

```python
embeddings = model.get_embeddings(sequences, show_progress=False)
```

## Available Models

List all registered models:

```python
from ablms import list_models

print(list_models())
# {'igbert': 'encoder', 'igt5': 'encoder', 'antiberta2': 'encoder',
#  'balm': 'encoder', 'antiberty': 'encoder', 'ablang2': 'encoder',
#  'ftesm': 'encoder', 'iglm': 'generative'}
```

## Notes on Specific Models

### IgT5

IgT5 is an encoder-only T5 model and does **not** have a masked language modeling head. Methods like `get_logits()`, `pseudo_log_likelihood()`, and `fill_mask()` will raise `UnsupportedOperationError`:

```python
from ablms import load_model

model = load_model("igt5")

# These work:
embeddings = model.get_embeddings(sequences)
attention = model.get_attention(sequences)

# These raise UnsupportedOperationError:
# model.get_logits(sequences)
# model.fill_mask(sequences)
```

### ft-ESM

ft-ESM is an ESM2-based model (finetuned from `facebook/esm2_t33_650M_UR50D`) optimized for paired antibody sequences. It uses a unique `<cls><cls>` separator (two consecutive CLS tokens) between chains:

```python
from ablms import load_model, AntibodySequence

model = load_model("ftesm")

# Paired sequences work well with ft-ESM
paired = AntibodySequence(
    heavy="EVQLVESGGGLVQPGRSLRLSCAASGFTFS",
    light="DIQMTQSPSSLSASVGDRVTITCRASQSIS"
)
embeddings = model.get_embeddings([paired])

# Single chain sequences also work
single = AntibodySequence(heavy="EVQLVESGGGLVQPGRSLRLSCAASGFTFS")
embeddings = model.get_embeddings([single])
```

### Single-Chain Models

AntiBERTy only supports single chain sequences. Passing paired sequences will raise `PairedSequenceError`:

```python
from ablms import load_model, AntibodySequence

model = load_model("antiberty")  # Single-chain only

# This works:
model.get_embeddings([AntibodySequence(heavy="EVQLVESGG...")])

# This raises PairedSequenceError:
# model.get_embeddings([AntibodySequence(heavy="...", light="...")])
```

## License

MIT License - see [LICENSE](LICENSE) for details.
