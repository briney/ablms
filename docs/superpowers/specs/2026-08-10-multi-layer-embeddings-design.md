# Multi-layer selection in `get_embeddings()`

**Date:** 2026-08-10
**Status:** Awaiting review

## Problem

`get_embeddings(layer=...)` extracts exactly one layer. A common analysis — the one ESM-C
popularised for UMAP/t-SNE projections — concatenates the CLS-pooled representation from
*every* layer into a single feature vector per sequence. Today that requires either N separate
calls to `get_embeddings()` (N full forward passes for data the model already computed) or
`get_hidden_states()`, which returns token-level output for all layers with no pooling and no
streaming.

`get_hidden_states()` is the unbounded-memory pattern that
[2026-08-09-embedding-memory-design.md](2026-08-09-embedding-memory-design.md) was written to
eliminate: it accumulates `[batch, seq_len, hidden_dim]` for every layer across the whole
dataset before returning anything.

### The relevant fact about the existing implementations

Every HuggingFace-backed encoder already runs its forward pass with `output_hidden_states=True`
and then discards all but one layer:

```python
# encoders/igbert.py:162-169, and identically in antiberta2, antiberty, igt5, esm2, balm, ftesm
outputs = self._model.bert(**tokenized, output_hidden_states=True)
hidden_states = outputs.hidden_states
embeddings = hidden_states[layer]
```

Multi-layer selection is therefore **free** for those models — the tensors are already
materialised on the device. AbLang2 is the one model that genuinely selects
(`return_rep_layers=[layer_idx]`, `ablang2.py:187`), and it accepts a list natively. AbLang v1
cannot produce intermediate layers at all and currently *silently ignores* the `layer`
argument (`ablang.py:353-355`).

## Goals

- `layer=` accepts a single index (unchanged), an explicit list, or `"all"`.
- Single-index calls are byte-identical to today, in both value and tensor shape.
- Pooled multi-layer runs stay memory-bounded and compose with `iter_embeddings()`.
- No per-encoder forward-pass changes. Only three encoders need any code at all: `IgT5` and
  `AbLang2` override how the layer count is read, and `AbLang` gets its guard.
- A model that cannot honour a layer request raises instead of returning the wrong layer.

## Non-goals

- Per-layer pooling strategies (e.g. CLS on layer 0, mean on layer 12). One strategy applies
  to all selected layers.
- Layer-weighted combination (learned scalar mix / "ESM-style layer weighting"). Callers can
  build it from a stacked output.
- Extending multi-layer selection to `get_attention()` or `get_logits()`.

## API

`layer: int | list[int] | Literal["all"] = -1`

The parameter keeps its singular name. Renaming it to `layers` would break keyword callers for
no functional gain, and `layer=[0, 6, 12]` reads acceptably. `"all"` is matched
case-insensitively.

The argument's *type* determines whether the output has a layer axis. `layer=-1` and
`layer=[-1]` return different shapes: the first is the single-layer form, the second is a
one-element stack. This is intentional — it means a caller building a layer list
programmatically gets a stable shape regardless of how many layers ended up in the list.

### Shapes

| Call | Result shape |
|---|---|
| `layer=-1` | `[B, S, D]` |
| `layer=-1, pooling="cls"` | `[B, D]` |
| `layer=[0, 6, 12]` | `[B, 3, S, D]` |
| `layer=[0, 6, 12], pooling="cls"` | `[B, 3, D]` |
| `layer="all"` | `[B, n_layers+1, S, D]` |
| `layer="all", pooling="cls"` | `[B, n_layers+1, D]` |

The layer axis is inserted immediately after the batch axis, in the order the layers were
requested. `"all"` is ascending from 0.

### The ESM-C recipe

```python
out = model.get_embeddings(sequences, layer="all", pooling="cls")
umap.UMAP().fit_transform(out.concat_layers().numpy())   # [B, (n_layers+1) * D]
```

### Layer indexing

Indices address `hidden_states`, matching what the code already does: index `0` is the
embedding-layer output and index `i` is the output of transformer block `i`. A model with
`num_layers = 12` therefore exposes 13 selectable indices, `0..12`, and negatives `-1..-13`.
This is the HuggingFace convention (`len(hidden_states) == config.num_hidden_layers + 1`) and
matches AbLang2's existing arithmetic at `ablang2.py:177`.

Validation, in `get_embeddings()` before any work is dispatched:

- Out-of-range index → `ValueError` naming the model and its valid range.
- Empty list → `ValueError`.
- Duplicate indices, after resolving negatives (e.g. `[12, -1]` on a 12-block model) →
  `ValueError`. Almost always a mistake, and cheap to catch.
- Order is preserved as given; the list is not sorted.

`EmbeddingOutput.layers` always holds resolved non-negative indices, so `layer=[-1]` on a
12-block model reports `layers == [12]`.

## Implementation

### Where the layer count comes from

Validation happens in the parent process, before dispatch, so the parent needs the layer count.
It has it: **every encoder loads its model eagerly in `__init__`** (`igbert.py:57`,
`esm2.py:85`, and the same line in all seven others), including in multi-GPU mode. `self._model`
is therefore always available by the time `get_embeddings()` is called, and the count can be
read from the live checkpoint rather than declared:

```python
class EncoderAbLM(BaseAbLM):
    supports_intermediate_layers: bool = True

    @property
    def num_layers(self) -> int:
        """Number of transformer blocks. Selectable indices are 0..num_layers."""
        return self._model.config.num_hidden_layers
```

Reading the checkpoint beats static metadata here because two encoders have no fixed depth:
`ESM2` selects among six variants from 6 to 48 layers via `model_id` (`esm2.py:17-22`), and
`FtESM` wraps a fine-tuned ESM-2 whose depth follows its base. A declared constant would have
to be maintained per variant and could silently disagree with the weights actually loaded.

Two encoders need an override because their model object is not a standard HF masked-LM:

- `IgT5` loads `T5EncoderModel` (`igt5.py:63`), whose `T5Config` spells the field `num_layers`,
  not `num_hidden_layers`.
- `AbLang2` has no HF config; it counts `len(self._model.encoder_blocks)`, which is the
  arithmetic already inlined at `ablang2.py:177`.

The value is cross-checked where the tensors are in hand: the selection helper raises if
`len(hidden_states) != num_layers + 1`. A `@pytest.mark.slow` test asserts the property against
a real forward pass for every installed model, so a wrong override surfaces loudly.

`CLAUDE.md`'s "Adding a New Encoder Model" checklist gains a note that `num_layers` must be
overridden if the model's config does not expose `num_hidden_layers`.

### Routing, without touching the encoders

`_forward_all_hidden_states()` is already implemented by every encoder. The multi-layer path
reuses it rather than widening `_forward_embeddings()`'s signature across eight files:

```python
# EncoderAbLM._process_embeddings_batch
if isinstance(layer, int):
    embeddings, mask = self._forward_embeddings(tokenized, layer)      # unchanged path
else:
    hidden_states, mask = self._forward_all_hidden_states(tokenized)
    selected = [hidden_states[i] for i in layer]                       # already resolved
```

For the HuggingFace models this is the identical forward pass at identical cost. For AbLang2 it
requests all rep layers where a subset would do — a small waste on partial selections,
accepted in exchange for zero per-encoder code.

### Pooling before stacking

Pooling applies to each selected layer *before* the layers are stacked:

```python
if pooling is not None:
    pooled = [apply_pooling(h, strategy=pooling, attention_mask=mask) for h in selected]
    embeddings = torch.stack(pooled, dim=1)        # [B, L, D]
    mask = None
else:
    embeddings = torch.stack(selected, dim=1)      # [B, L, S, D]
```

`[B, L, S, D]` is never materialised on the pooled path. This preserves the "reduce before
transfer" invariant the memory work established: the queue payload for
`layer="all", pooling="cls"` at `batch_size=32`, `D=1024`, 13 layers is **1.7 MB per batch**.
The layer stack itself adds nothing on the device — `output_hidden_states=True` had already
allocated every layer.

### Executor padding must stop assuming dim 1 is the sequence axis

`MultiGPUExecutor._pad_tensors_to_max_length` (`executor.py:512-548`) pads dim 1 to the maximum
across batches, because batches are tokenized independently and get different padded lengths.
It hardcodes the 3D `[B, S, D]` and 2D `[B, S]` cases and falls through to a 2D padding tensor
for anything else. A token-level multi-layer result is 4D `[B, L, S, D]`, so it would build a
`[B, pad]` tensor and `torch.cat` would raise a dimension-mismatch `RuntimeError` — and only on
datasets whose batches happen to have different padded lengths, which is most of them.

Replace the hardcoded cases with the general rule: **all tensors in a group have identical
shape except along the axis that varies; find the differing axes (excluding dim 0, which is the
concatenation axis) and pad them to the maximum with `torch.nn.functional.pad`.** If no axis
differs, return unchanged.

This is correct for every current caller without a new parameter, and it removes a second
latent bug of the same kind: `get_attention()` returns `[B, L, H, S, S]`, which hits the same
fall-through today.

## `EmbeddingOutput` changes

New field `layers: list[int] | None = None`. `None` means single-layer, and `layer` carries the
index as before. When `layers` is set, `layer` is set to `None` — widening it to
`int | None` — rather than holding a misleading scalar.

```python
@property
def is_multi_layer(self) -> bool: ...          # layers is not None

@property
def num_layers(self) -> int: ...               # len(layers), or 1

@property
def is_pooled(self) -> bool:
    return self.embeddings.ndim == (3 if self.is_multi_layer else 2)

def get_layer(self, layer: int) -> torch.Tensor:
    """Slice out one layer by its original index. Raises if not selected."""

def concat_layers(self) -> torch.Tensor:
    """Fold the layer axis into the hidden axis: [B, L, D] -> [B, L*D],
    [B, L, S, D] -> [B, S, L*D]. Raises on single-layer output."""
```

`get_layer()` takes the *model's* layer index (`get_layer(6)`), not a position in the stack,
and raises `ValueError` if that layer was not selected.

The shape-derived `is_pooled` keeps the existing single-layer behaviour exactly, so
`tests/test_outputs.py:26` and `:33` — which construct bare 2D and 3D tensors with no `pooled=`
argument — continue to pass unchanged.

Three methods index dimension 1 assuming it is the sequence axis and need a multi-layer branch:
`get_chain_embeddings()`, `get_sequence_tokens()` (and therefore `__getitem__`/`__iter__`).
Under multi-layer they return a leading layer axis: `[L, chain_len, D]` and
`[L, actual_seq_len, D]`. `to()`, `cpu()`, and `numpy()` propagate the new field. `__repr__`
prints `layers=[...]` when multi-layer.

## `get_hidden_states()`

Kept, with its `list[EmbeddingOutput]` return type, reimplemented as a wrapper:

```python
out = self.get_embeddings(sequences, layer="all", batch_size=..., show_progress=...)
return [ ... one EmbeddingOutput per layer, sliced off the stack ... ]
```

One code path, no breakage for existing callers, and it inherits the bounded-memory
improvements. Its docstring gains a pointer to `layer="all"` as the preferred form and a note
that it materialises the full token-level output for every layer.

## `iter_embeddings()`

Gets the same `layer` parameter and the same validation. It already forwards `layer` through to
`_process_embeddings_batch` (`encoder.py:220-228`), so the streaming path needs only the
parameter type widened and `layers=` set on each yielded `EmbeddingOutput`.

This is the documented answer for token-level multi-layer runs, which are genuinely large:
`layer="all"` at `batch_size=32`, padded length 256, `D=1024`, 13 layers is **436 MB per
batch**, against 33.5 MB for a single layer. `get_embeddings()` will not refuse it — a small
dataset is a legitimate use — but both the docstring and the README steer token-level
all-layer work to `iter_embeddings()`.

## AbLang v1

`AbLang` sets `supports_intermediate_layers = False`. Validation raises
`UnsupportedOperationError` for any request other than the final layer: `"all"`, any
multi-element list, and any single index that does not resolve to the last layer.

This is a deliberate behaviour change. Today `model.get_embeddings(seqs, layer=3)` on AbLang
returns the *final* layer with no indication that the request was ignored — a silent wrong
answer. `layer=-1` and the default keep working.

AbLang also overrides `num_layers` with a literal `12` (AbRep's depth), because its model object
has no HF config for the base property to read. Only the final layer is reachable, so the value
affects nothing but which explicit positive index is accepted as "final" and the wording of the
error. It should be confirmed against a real forward pass whenever the `ablang` package is
installed.

Because `get_hidden_states()` is reimplemented over `layer="all"`, it would start raising on
AbLang — which today returns a single-element list. It therefore selects
`"all" if self.supports_intermediate_layers else -1`, wrapping the single-layer result in a
one-element list. AbLang's `get_hidden_states()` still returns a one-element list holding its
final layer. One field does change: that output's `layer` is now `-1` rather than `0`. The old
value came from enumerating a one-element list, and `0` denotes the embedding layer in this
library's convention, so it mislabelled AbLang's final layer; `-1` matches what
`get_embeddings()` reports for the same model.

## Error handling

| Condition | Exception |
|---|---|
| Index outside `-(num_layers+1) .. num_layers` | `ValueError` |
| Empty list | `ValueError` |
| Duplicate indices after resolution | `ValueError` |
| `layer` is a string other than `"all"` | `ValueError` |
| Non-final request on a model with `supports_intermediate_layers = False` | `UnsupportedOperationError` |
| `concat_layers()` / `get_layer()` on single-layer output | `ValueError` |
| `get_layer()` for a layer that was not selected | `ValueError` |
| `num_layers` disagrees with `len(hidden_states)` | `RuntimeError` |

All argument validation happens in `get_embeddings()`/`iter_embeddings()` before dispatch, so
bad input fails at the call site rather than inside a worker — matching the eager-validation
pattern already used for sequences (`encoder.py:194-197`).

## Testing

Shape and back-compat, on a small real model:

- `layer=-1` output is identical in shape, dtype, and value to the pre-change implementation,
  pooled and unpooled.
- `layer=[-1]` equals `layer=-1` after squeezing the layer axis.
- `layer=[0, 6, 12]` gives `[B, 3, S, D]`; each slice equals a separate `layer=i` call.
- `layer="all"` gives `n_layers + 1` layers in ascending order.
- `concat_layers()` on `[B, L, D]` matches `torch.cat([...], dim=-1)` over `get_layer()` calls.
- `get_layer(6)` returns the same tensor as `get_embeddings(layer=6)`.

Memory and parallel behaviour, extending `tests/test_embedding_memory.py`:

- Multi-layer pooled batches cross the queue at `[B, L, D]`, not `[B, L, S, D]` — the existing
  payload-shape assertions extended to the layer axis.
- Multi-GPU multi-layer with **deliberately unequal sequence lengths across batches**, which is
  what exercises the rewritten padding. This is the test that would have caught the 4D
  fall-through.
- `iter_embeddings(layer="all")` yields per-batch outputs carrying correct `layers`.

Validation: each error-table row gets a case. Duplicate detection is tested through the
negative-index form (`[12, -1]`), since that is the one a caller writes by accident.

A `@pytest.mark.slow` test verifies `num_layers + 1 == len(hidden_states)` from a real forward
pass, parametrized over every encoder whose package is installed and skipping the rest. This is
what catches a missing `num_layers` override on a newly added encoder.

## Known risks

**A new encoder whose config lacks `num_hidden_layers` fails at inference, not at import.** The
base property raises `AttributeError` on first use rather than when the class is defined. The
slow test covers this for installed models; an encoder whose package is absent from the test
environment (AbLang today) is not covered until someone runs it.

**AbLang2 over-computes on partial selections.** `layer=[0, 6]` requests all 13 rep layers and
discards 11. Fixable later by widening `_forward_all_hidden_states` with an optional layer
subset; not worth per-encoder divergence now.

**The padding rewrite touches a shared path.** Every executor caller — embeddings, logits,
attention, mask scan — goes through `_pad_tensors_to_max_length`. The new rule is a superset of
the old behaviour for 2D and 3D inputs, but it is the highest-risk edit in this change, which
is why the unequal-length multi-GPU test is called out explicitly above.
