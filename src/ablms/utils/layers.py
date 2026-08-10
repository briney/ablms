"""Resolution and validation of layer selections for embedding extraction."""

from __future__ import annotations

from ablms.exceptions import UnsupportedOperationError

ALL_LAYERS = "all"


def resolve_layer_selection(
    layer: int | list[int] | str,
    num_layers: int,
    *,
    model_name: str,
    supports_intermediate_layers: bool = True,
) -> int | list[int]:
    """
    Validate a layer selection and resolve it to concrete indices.

    Indices address the model's hidden_states tuple: index 0 is the
    embedding-layer output and index i is the output of transformer block i, so
    a model with `num_layers` blocks has `num_layers + 1` selectable indices.

    A single int is returned unchanged rather than normalized, so a default
    `layer=-1` call still reports `layer=-1` on its output, exactly as before
    multi-layer support existed. Lists are always resolved to non-negative
    indices, because they are used to index hidden_states directly and are
    reported on `EmbeddingOutput.layers`.

    Args:
        layer: A single index, a list/tuple/range of indices, or "all"
            (case-insensitive) for every layer in ascending order.
        num_layers: Count of transformer blocks in the model.
        model_name: Model name, used in error messages.
        supports_intermediate_layers: False for models that can only produce
            their final layer.

    Returns:
        The int unchanged for a single-layer selection, or a list of resolved
        non-negative indices in the order requested.

    Raises:
        ValueError: If the selection is malformed, empty, out of range, or
            contains duplicates after negative indices are resolved.
        UnsupportedOperationError: If a non-final layer is requested from a
            model with supports_intermediate_layers=False.
    """
    n_selectable = num_layers + 1

    if isinstance(layer, str):
        if layer.lower() != ALL_LAYERS:
            raise ValueError(
                f"Invalid layer selection {layer!r}. Pass an int, a list of "
                f"ints, or {ALL_LAYERS!r}."
            )
        resolved = list(range(n_selectable))
        _check_supported(
            resolved, n_selectable, model_name, supports_intermediate_layers
        )
        return resolved

    if isinstance(layer, (list, tuple, range)):
        indices = list(layer)
        if not indices:
            raise ValueError(
                "Layer selection list is empty. Pass at least one index, or "
                f"{ALL_LAYERS!r} for every layer."
            )
        for index in indices:
            _validate_index(index, n_selectable, model_name)

        resolved = [_normalize(index, n_selectable) for index in indices]
        _reject_duplicates(resolved, indices, model_name)
        _check_supported(
            resolved, n_selectable, model_name, supports_intermediate_layers
        )
        return resolved

    _validate_index(layer, n_selectable, model_name)
    _check_supported(
        [_normalize(layer, n_selectable)],
        n_selectable,
        model_name,
        supports_intermediate_layers,
    )
    return layer


def _validate_index(index: object, n_selectable: int, model_name: str) -> None:
    """Reject non-ints and out-of-range indices."""
    # bool is a subclass of int; True must not silently mean layer 1.
    if not isinstance(index, int) or isinstance(index, bool):
        raise ValueError(
            f"Layer indices must be ints, got {index!r} " f"({type(index).__name__})."
        )
    if not -n_selectable <= index < n_selectable:
        raise ValueError(
            f"Layer {index} is out of range for {model_name}: valid indices are "
            f"0..{n_selectable - 1} or -1..-{n_selectable}. Index 0 is the "
            f"embedding layer and {n_selectable - 1} is the final block."
        )


def _normalize(index: int, n_selectable: int) -> int:
    """Convert a negative index to its non-negative equivalent."""
    return index + n_selectable if index < 0 else index


def _reject_duplicates(
    resolved: list[int], requested: list[int], model_name: str
) -> None:
    """Reject repeated layers, including via mixed negative and positive forms."""
    seen: set[int] = set()
    for position, index in enumerate(resolved):
        if index in seen:
            raise ValueError(
                f"Duplicate layer {index} in selection {requested} for "
                f"{model_name} (entry {requested[position]!r} repeats an "
                f"earlier layer). Each layer may be selected only once."
            )
        seen.add(index)


def _check_supported(
    resolved: list[int],
    n_selectable: int,
    model_name: str,
    supports_intermediate_layers: bool,
) -> None:
    """Reject non-final selections for models that expose only their last layer."""
    if supports_intermediate_layers:
        return

    final = n_selectable - 1
    if resolved != [final]:
        raise UnsupportedOperationError(
            f"{model_name} exposes only its final layer ({final}); requested "
            f"{resolved}. Use layer=-1, or omit the argument."
        )
