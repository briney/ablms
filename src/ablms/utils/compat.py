"""Compatibility shims for third-party backends under transformers 5.x.

Two of the model packages this library wraps, `iglm` and `antiberty`, were
written against the transformers 4.x API and break under 5.x. All three of the
underlying bugs are upstream and one line each; the shims here repair them in
place so those models stay usable without pinning the whole project to
transformers 4, which the eight other models have no need of. See issue #5.

The bugs:

1. `BertTokenizerFast(vocab_file=...)` / `BertTokenizer(vocab_file=...)`.
   In transformers 5 the first parameter was renamed `vocab_file` -> `vocab`,
   and `BertTokenizerFast` became an alias for `BertTokenizer`. The old keyword
   is swallowed by `**kwargs` with no error, `vocab` defaults to `None`, and the
   tokenizer is built holding only its five special tokens. Every residue and
   control token then resolves to `[UNK]`. This one is silent: the model still
   runs and still returns correctly shaped output, it is simply meaningless.
   Affects `iglm/model/IgLM.py` and `antiberty/AntiBERTyRunner.py`.

2. `self.init_weights()` in a `PreTrainedModel.__init__`. Transformers >=4.6
   replaced this with `self.post_init()`, and in 5.x `post_init()` is the only
   thing that populates `all_tied_weights_keys`; `init_weights()` merely
   consumes it. Loading raises `AttributeError` from `from_pretrained`.
   Affects `antiberty/model/AntiBERTy.py`.

Every shim is conditional on detecting the actual symptom rather than on a
version check. That means each becomes a no-op the moment upstream fixes its
bug, and each works unchanged on transformers 4.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ensure_post_init_called", "repair_bert_tokenizer"]

_PATCH_MARKER = "_ablms_post_init_shim"


def repair_bert_tokenizer(
    tokenizer: Any,
    vocab_file: str,
    probe_tokens: list[str],
    **init_kwargs: Any,
) -> Any:
    """
    Return a tokenizer whose vocabulary actually loaded.

    Checks whether `probe_tokens` resolve to real ids. If they collapse to the
    unknown-token id, the vocabulary never loaded (bug 1 above) and the
    tokenizer is rebuilt using the `vocab=` keyword that transformers 5
    expects. An already-healthy tokenizer is returned untouched.

    Args:
        tokenizer: The tokenizer the third-party package constructed.
        vocab_file: Path to the vocabulary file it was meant to load.
        probe_tokens: Tokens that must exist in a correctly loaded vocabulary.
            Use tokens the model genuinely depends on, e.g. IgLM's `[HEAVY]`
            or a couple of amino acids.
        **init_kwargs: Passed to the rebuilt tokenizer, e.g.
            `do_lower_case=False`. Must match how the package built it.

    Returns:
        Either the original tokenizer, or a rebuilt one of the same class.

    Raises:
        ModelLoadError: If rebuilding does not resolve the probe tokens, which
            means the cause is something other than the known keyword rename.
    """
    from ablms.exceptions import ModelLoadError

    if not _tokenizer_is_degenerate(tokenizer, probe_tokens):
        return tokenizer

    # Same class, correct keyword. `type(tokenizer)` rather than a hardcoded
    # class so this keeps working if a package swaps its tokenizer type.
    repaired = type(tokenizer)(vocab=vocab_file, **init_kwargs)

    if _tokenizer_is_degenerate(repaired, probe_tokens):
        raise ModelLoadError(
            f"Could not load a usable vocabulary from {vocab_file}. Rebuilding "
            f"with `vocab=` still leaves {probe_tokens} unresolved, so this is "
            "not the known transformers 5 keyword rename. See issue #5."
        )
    return repaired


def _tokenizer_is_degenerate(tokenizer: Any, probe_tokens: list[str]) -> bool:
    """True if any probe token resolves to the unknown-token id."""
    unk_id = tokenizer.unk_token_id
    if unk_id is None:
        return False
    return any(tokenizer.convert_tokens_to_ids(t) == unk_id for t in probe_tokens)


def ensure_post_init_called(model_cls: type) -> None:
    """
    Make a `PreTrainedModel` subclass call `post_init()` during construction.

    Repairs bug 2 above by wrapping `__init__` to call `post_init()` when the
    original did not leave `all_tied_weights_keys` behind. Idempotent: calling
    this repeatedly wraps the class only once, and the wrapper does nothing on
    a class that already initialises correctly.

    Args:
        model_cls: The `PreTrainedModel` subclass to patch, e.g. AntiBERTy's
            model class. Patched in place because the failure happens inside
            `from_pretrained`, before any instance is available to us.
    """
    if getattr(model_cls, _PATCH_MARKER, False):
        return

    original_init = model_cls.__init__

    def _init_with_post_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        # Guard rather than always calling: a fixed upstream, or a transformers
        # version that populates this elsewhere, must not be double-initialised.
        if not hasattr(self, "all_tied_weights_keys"):
            self.post_init()

    # `setattr` rather than direct assignment: patching a class's `__init__` is
    # deliberately dynamic, and assigning to it directly does not type-check
    # against `type.__init__`'s declared signature.
    setattr(model_cls, "__init__", _init_with_post_init)  # noqa: B010
    setattr(model_cls, _PATCH_MARKER, True)
