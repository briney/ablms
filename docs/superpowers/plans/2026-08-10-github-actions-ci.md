# GitHub Actions CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a blocking GitHub Actions CI workflow running ruff, black, ty, and the non-slow pytest suite, after cleaning the repository so all four checks pass on the first run.

**Architecture:** One workflow file with four jobs running in parallel (`lint`, `typecheck`, `test` across three Python versions, and a non-blocking `smoke` job). Six preparatory tasks bring the repository to a clean state before the workflow lands, because a gate that is red on day one cannot be made a required check.

**Tech Stack:** GitHub Actions, `actions/setup-python@v5`, ty 0.0.40, ruff 0.16.2, black 26.5.1, pytest, PyTorch CPU wheels.

**Spec:** `docs/superpowers/specs/2026-08-10-github-actions-ci-design.md`

## Global Constraints

- **Type hints:** Python 3.10+ union syntax. Write `x | None`, never `Optional[x]`. Never import `Optional` or `Union` from `typing`.
- **Line length:** 88 (`black` and `ruff` are both configured to this in `pyproject.toml`).
- **Supported Python:** `requires-python = ">=3.10"`; classifiers list 3.10, 3.11, 3.12. ty infers `--python-version 3.10` from `requires-python`.
- **Tool versions are pinned exactly:** `ty==0.0.40`, `ruff==0.16.2`, `black==26.5.1`. Runtime dependencies stay unpinned.
- **Run pytest as `python -m pytest`**, never bare `pytest` — bare `pytest` resolves to a different interpreter in this environment and fakes a total collection failure.
- **ty gate scope is `src/` only.** Do not add `tests/` to the ty command; `tests/` has 73 further diagnostics that are explicitly out of scope.
- **Never disable a ty or ruff rule globally to make an error go away.** Two exceptions are authorised by the spec and named explicitly in Tasks 2 and 4: the `N812` ruff ignore, and one `# ty: ignore[unresolved-import]` in `encoders/ablang.py`.
- **Do not attempt to make IgLM or AntiBERTy actually run.** They are broken under transformers 5.x. Task 5 fixes types and API shape only.
- **Verification baseline:** at the end of Task 7, `ruff check src/ tests/`, `black --check src/ tests/`, `ty check src/`, and `python -m pytest -m "not slow"` must all pass.
- **`tests/test_smoke.py` is expected to FAIL when committed, by design.** IgLM and AntiBERTy are broken under transformers 5.x; these tests exist to report when that is fixed, and the CI job running them is `continue-on-error: true`. Committing them red is intentional, not an oversight — do not delete, skip, or `xfail` them to make the suite green. They are excluded from the blocking `test` job because `smoke` implies `slow`.
- **Three `Any` annotations and one `# ty: ignore` are deliberate**, each with a comment stating why: HuggingFace model/tokenizer objects are dynamically shaped, and `ablang` v1 is genuinely not a declared dependency. These are the spec's authorised escapes, not shortcuts.

## Baseline numbers (measured 2026-08-10)

| Check | Before | After this plan |
| --- | --- | --- |
| `ruff check src/ tests/` | 57 errors | 0 |
| `black --check src/ tests/` | 18 files | 0 |
| `ty check src/` | 204 diagnostics | 0 |
| `python -m pytest -m "not slow"` | 253 passed, 16 skipped | passing, plus new tests |

---

### Task 1: Align local tool versions with the pins

Formatting with a different tool version than CI enforces is how a `--check` gate fails on a clean checkout. The local environment currently has ruff 0.15.6; the pin will be 0.16.2.

**Files:**
- No source changes. Environment only.

**Interfaces:**
- Consumes: nothing.
- Produces: an environment where `ruff --version` is 0.16.2, `black --version` is 26.5.1, `ty --version` is 0.0.40. Every later task depends on these exact versions.

- [ ] **Step 1: Install the pinned tool versions**

```bash
pip install 'ty==0.0.40' 'ruff==0.16.2' 'black==26.5.1'
```

- [ ] **Step 2: Confirm the versions**

```bash
ty --version && ruff --version && black --version
```

Expected output contains exactly `ty 0.0.40`, `ruff 0.16.2`, `black, 26.5.1`.

- [ ] **Step 3: Record the baseline**

```bash
ruff check src/ tests/ 2>&1 | tail -2
black --check src/ tests/ 2>&1 | tail -1
ty check src/ 2>&1 | tail -1
```

Expected: `Found 57 errors.`, `18 files would be reformatted, 31 files would be left unchanged.`, `Found 204 diagnostics`.

If any number differs, stop and report it rather than proceeding — the rest of the plan is written against these numbers.

---

### Task 2: Make ruff and black clean

**Files:**
- Modify: `pyproject.toml` (add `N812` to `[tool.ruff.lint] ignore`)
- Modify: 18 files reformatted by black, ~30 files touched by ruff's import sorting and unused-import removal
- Modify by hand: `src/ablms/encoders/antiberta2.py:152`, `src/ablms/encoders/antiberty.py:252`, `src/ablms/encoders/igbert.py:122`, `src/ablms/generators/iglm.py:176-177`, `src/ablms/utils/pooling.py:109`

**Interfaces:**
- Consumes: pinned tool versions from Task 1.
- Produces: `ruff check src/ tests/` and `black --check src/ tests/` both exit 0. No API changes.

- [ ] **Step 1: Add the `N812` ignore to pyproject.toml**

Ten files trigger `N812 Lowercase functional imported as non-lowercase F` for `import torch.nn.functional as F`. `F` is the universal PyTorch convention and renaming it would make the codebase worse, so the rule is ignored rather than obeyed.

In `pyproject.toml`, change:

```toml
[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
ignore = ["E501"]
```

to:

```toml
[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]
# E501: line length is enforced by the formatter, not the linter.
# N812: `import torch.nn.functional as F` is the universal PyTorch convention.
ignore = ["E501", "N812"]
```

- [ ] **Step 2: Verify the N812 errors are gone and count what remains**

```bash
ruff check src/ tests/ 2>&1 | tail -2
```

Expected: `Found 47 errors.` (57 minus the 10 `N812`).

- [ ] **Step 3: Apply ruff's automatic fixes**

```bash
ruff check --fix src/ tests/
ruff check src/ tests/ 2>&1 | tail -3
```

Expected: 41 fixed, leaving `Found 6 errors.` — all `F841` unused-local-variable.

- [ ] **Step 4: Inspect the 6 remaining F841 sites before changing them**

```bash
ruff check src/ tests/ --output-format concise
```

Expected exactly these six:

```
src/ablms/encoders/antiberta2.py:152:9: F841 Local variable `cls_token_id` is assigned to but never used
src/ablms/encoders/antiberty.py:252:21: F841 Local variable `token` is assigned to but never used
src/ablms/encoders/igbert.py:122:9: F841 Local variable `cls_token_id` is assigned to but never used
src/ablms/generators/iglm.py:176:13: F841 Local variable `prefix` is assigned to but never used
src/ablms/generators/iglm.py:177:13: F841 Local variable `suffix` is assigned to but never used
src/ablms/utils/pooling.py:109:5: F841 Local variable `batch_size` is assigned to but never used
```

Read each site. These are dead assignments, but confirm that the value is genuinely unused rather than that a nearby line should have been using it — an unused `cls_token_id` next to code that handles `sep_token_id` may be a symptom of a real omission. If any site looks like a latent bug rather than dead code, stop and report it instead of deleting the line.

- [ ] **Step 5: Delete all six dead assignments**

Delete the assignment line at each of `antiberta2.py:152`, `antiberty.py:252`, `igbert.py:122`, `pooling.py:109`, and both `iglm.py:176-177` (`prefix` and `suffix`). Task 5 rewrites that `iglm.py` block anyway, but removing them here keeps this task's verification self-consistent: ruff must be fully clean at the end of Task 2, not carrying two known failures across tasks.

For example, in `src/ablms/encoders/igbert.py`, this becomes a single line:

```python
        sep_token_id = self._tokenizer.sep_token_id
        cls_token_id = self._tokenizer.cls_token_id
```

```python
        sep_token_id = self._tokenizer.sep_token_id
```

- [ ] **Step 6: Run black**

```bash
black src/ tests/
```

Expected: `18 files reformatted` (the count may differ slightly now that ruff has rewritten import blocks; any number is fine as long as Step 7 passes).

- [ ] **Step 7: Verify lint and format are clean, and tests still pass**

```bash
ruff check src/ tests/
black --check src/ tests/
python -m pytest -m "not slow" -q 2>&1 | tail -3
```

Expected: `All checks passed!`, `All done!`, and `253 passed, 16 skipped, 139 deselected`.

Both tools must be fully clean before committing. If ruff still reports anything, do not proceed.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/ tests/
git commit -m "Format with black and fix ruff lint errors"
```

---

### Task 3: Annotate the model attributes

This is the single highest-leverage change in the plan: it takes ty from 204 diagnostics to 46.

**Files:**
- Modify: `src/ablms/core/base.py:83-84`

**Interfaces:**
- Consumes: nothing.
- Produces: `BaseAbLM._model: Any` and `BaseAbLM._tokenizer: Any`, which every encoder subclass relies on for the rest of the type fixes.

- [ ] **Step 1: Confirm `Any` is already imported**

```bash
grep -n "^from typing" src/ablms/core/base.py
```

Expected: `from typing import Any, TYPE_CHECKING` — `Any` is already imported, so no import change is needed.

- [ ] **Step 2: Annotate both attributes**

In `src/ablms/core/base.py`, change:

```python
        self._model = None
        self._tokenizer = None
```

to:

```python
        # Deliberately `Any`: concrete model and tokenizer types come from
        # HuggingFace and are dynamically shaped. Subclasses reach for
        # attributes like `_model.bert` and `_tokenizer.sep_token_id` that no
        # static `PreTrainedModel` type declares.
        self._model: Any = None
        self._tokenizer: Any = None
```

- [ ] **Step 3: Verify the diagnostic count drops**

```bash
ty check src/ 2>&1 | tail -1
```

Expected: `Found 44 diagnostics` (down from 204).

Note: 44, not 46. Task 2's deletion of the dead `prefix` and `suffix` locals in `generators/iglm.py` removed two `not-subscriptable` diagnostics along with the dead code, so `iglm.py` now carries 13 rather than 15. Task 5's end state of 28 is unaffected.

- [ ] **Step 4: Verify tests still pass**

```bash
python -m pytest -m "not slow" -q 2>&1 | tail -2
```

Expected: `253 passed, 16 skipped, 139 deselected`.

- [ ] **Step 5: Commit**

```bash
git add src/ablms/core/base.py
git commit -m "Annotate model and tokenizer attributes as Any"
```

---

### Task 4: Add `AntibodySequence.primary_chain` and use it

Six call sites do `seq.heavy_chain or seq.light_chain`, producing `str | None`, and then immediately use the result as a `str`. `AntibodySequence.__init__` already guarantees at least one chain is set (`core/sequence.py:97-98` raises `InvalidSequenceError` otherwise), so the invariant exists but is not expressed in the types. One property fixes all six sites.

**Files:**
- Modify: `src/ablms/core/sequence.py` (add `primary_chain` property)
- Modify: `src/ablms/encoders/esm2.py:110`, `src/ablms/encoders/antiberty.py:83`, `src/ablms/encoders/ablang.py:130`
- Test: `tests/test_sequence.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `AntibodySequence.primary_chain -> str`. Task 5 also uses it at three sites in `generators/iglm.py`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sequence.py`:

```python
class TestPrimaryChain:
    def test_returns_heavy_when_only_heavy(self):
        seq = AntibodySequence(heavy="EVQLVESGGG")
        assert seq.primary_chain == "EVQLVESGGG"

    def test_returns_light_when_only_light(self):
        seq = AntibodySequence(light="DIQMTQSPSS")
        assert seq.primary_chain == "DIQMTQSPSS"

    def test_prefers_heavy_when_paired(self):
        seq = AntibodySequence(heavy="EVQLVESGGG", light="DIQMTQSPSS")
        assert seq.primary_chain == "EVQLVESGGG"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_sequence.py::TestPrimaryChain -q
```

Expected: FAIL with `AttributeError: 'AntibodySequence' object has no attribute 'primary_chain'`.

- [ ] **Step 3: Add the property**

In `src/ablms/core/sequence.py`, add alongside the other properties (near `is_paired` at line 127):

```python
    @property
    def primary_chain(self) -> str:
        """
        The heavy chain if present, otherwise the light chain.

        Single-chain models use this to pick the one sequence they operate on.
        `__init__` rejects a sequence with neither chain, so this always
        returns a string.

        Returns:
            The heavy chain sequence, or the light chain if there is no heavy.

        Raises:
            InvalidSequenceError: If neither chain is set, which `__init__`
                should already have prevented.
        """
        if self.heavy_chain is not None:
            return self.heavy_chain
        if self.light_chain is not None:
            return self.light_chain
        raise InvalidSequenceError("AntibodySequence has neither chain set")
```

`InvalidSequenceError` is already imported in this module — confirm with `grep -n InvalidSequenceError src/ablms/core/sequence.py`.

- [ ] **Step 4: Run the test to verify it passes**

```bash
python -m pytest tests/test_sequence.py::TestPrimaryChain -q
```

Expected: `3 passed`.

- [ ] **Step 5: Replace the three encoder call sites**

In `src/ablms/encoders/esm2.py`, inside `_format_for_model`:

```python
            sequence = seq.heavy_chain or seq.light_chain
```

becomes:

```python
            sequence = seq.primary_chain
```

Make the identical change in `src/ablms/encoders/antiberty.py:83` and `src/ablms/encoders/ablang.py:130`. All three currently read `sequence = seq.heavy_chain or seq.light_chain`.

- [ ] **Step 6: Verify three ty diagnostics are gone**

```bash
ty check src/ --output-format concise 2>&1 | grep -c 'error'
ty check src/ --output-format concise 2>&1 | grep 'replace' || echo "no replace errors remain"
```

Expected: count is `41` (44 minus 3). The `esm2.py`, `antiberty.py`, and `ablang.py` `Attribute 'replace' is not defined on 'None'` errors are gone. One `replace` error remains in `generators/iglm.py:218`; Task 5 fixes it.

- [ ] **Step 7: Run the full non-slow suite**

```bash
python -m pytest -m "not slow" -q 2>&1 | tail -2
```

Expected: `256 passed, 16 skipped, 139 deselected`.

- [ ] **Step 8: Commit**

```bash
git add src/ablms/core/sequence.py src/ablms/encoders/ tests/test_sequence.py
git commit -m "Add AntibodySequence.primary_chain and use it in encoders"
```

---

### Task 5: Fix the IgLM wrapper at the type and API level

The wrapper calls `iglm` with keyword names that do not exist, token values that are not in the vocabulary, and unpacks tuples from methods that return lists. **Do not try to make the model actually run** — it is broken under transformers 5.x independently of this code (see the spec). Fix the call shape and prove it with contract tests.

The real signatures, from the installed `iglm` package:

```python
generate(chain_token, species_token, prompt_sequence=None, num_to_generate=1000, top_p=1, temperature=1) -> list[str]
infill(sequence, chain_token, species_token, infill_range, num_to_generate=1000, top_p=1, temperature=1) -> list[str]
log_likelihood(sequence, chain_token, species_token, infill_range=None) -> float
```

**Files:**
- Create: `tests/test_iglm.py`
- Modify: `src/ablms/generators/iglm.py`

**Interfaces:**
- Consumes: `AntibodySequence.primary_chain` from Task 4.
- Produces: `SPECIES_MAP` and `CHAIN_TYPE_MAP` mapping to bracketed vocabulary tokens. No public API change to the `IgLM` class.

- [ ] **Step 1: Write the failing contract test**

Create `tests/test_iglm.py`:

```python
"""Contract tests for the IgLM wrapper's use of the `iglm` package API.

These do not instantiate the model. They check that the wrapper's calls match
the installed package's signatures and vocabulary, which is enough to catch
upstream API drift and needs no weights, network, or GPU.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from ablms.core.sequence import ChainType, Species
from ablms.generators.iglm import CHAIN_TYPE_MAP, SPECIES_MAP


@pytest.fixture(scope="session")
def iglm_vocab() -> set[str]:
    """Every token in the installed iglm package's vocabulary."""
    iglm_model = pytest.importorskip("iglm.model.IgLM")
    return set(Path(iglm_model.VOCAB_FILE).read_text().split())


@pytest.fixture(scope="session")
def iglm_class():
    """The installed IgLM class, without instantiating it."""
    return pytest.importorskip("iglm").IgLM


class TestTokenMaps:
    def test_species_map_values_are_vocab_tokens(self, iglm_vocab):
        for species, token in SPECIES_MAP.items():
            assert token in iglm_vocab, f"{species} maps to {token!r}, not in vocab"

    def test_chain_map_values_are_vocab_tokens(self, iglm_vocab):
        for chain, token in CHAIN_TYPE_MAP.items():
            assert token in iglm_vocab, f"{chain} maps to {token!r}, not in vocab"

    def test_every_enum_member_is_mapped(self):
        assert set(SPECIES_MAP) == set(Species)
        assert set(CHAIN_TYPE_MAP) == set(ChainType)


class TestCallSignatures:
    """The wrapper's keyword arguments must bind against the real signatures."""

    def test_generate_kwargs_bind(self, iglm_class):
        sig = inspect.signature(iglm_class.generate)
        sig.bind(
            None,  # self
            chain_token="[HEAVY]",
            species_token="[HUMAN]",
            prompt_sequence=None,
            num_to_generate=1,
            top_p=1.0,
            temperature=1.0,
        )

    def test_infill_kwargs_bind(self, iglm_class):
        sig = inspect.signature(iglm_class.infill)
        sig.bind(
            None,  # self
            sequence="EVQL",
            chain_token="[HEAVY]",
            species_token="[HUMAN]",
            infill_range=(1, 2),
            num_to_generate=1,
            temperature=1.0,
        )

    def test_log_likelihood_kwargs_bind(self, iglm_class):
        sig = inspect.signature(iglm_class.log_likelihood)
        sig.bind(
            None,  # self
            sequence="EVQL",
            chain_token="[HEAVY]",
            species_token="[HUMAN]",
        )

    @pytest.mark.parametrize("method", ["generate", "infill", "log_likelihood"])
    def test_wrapper_does_not_pass_our_enum_names(self, iglm_class, method):
        """`chain_type` and `species` are our names; iglm uses `*_token`."""
        params = inspect.signature(getattr(iglm_class, method)).parameters
        assert "chain_token" in params
        assert "species_token" in params
        assert "chain_type" not in params
        assert "species" not in params
```

- [ ] **Step 2: Run the test to verify the token-map tests fail**

```bash
python -m pytest tests/test_iglm.py -q
```

Expected: the two `TestTokenMaps` vocabulary tests FAIL (`HUMAN maps to 'human', not in vocab`). The `TestCallSignatures` tests pass already, since they describe the real API rather than the wrapper.

- [ ] **Step 3: Fix the token maps**

In `src/ablms/generators/iglm.py`, replace both maps. IgLM's `vocab.txt` defines exactly `[CAMEL]`, `[HUMAN]`, `[MOUSE]`, `[RABBIT]`, `[RAT]`, `[RHESUS]`, `[HEAVY]`, `[LIGHT]`.

```python
# Mapping from our Species enum to IgLM species control tokens. These are fed
# to IgLM's tokenizer, which asserts that every token is in its vocabulary.
SPECIES_MAP = {
    Species.HUMAN: "[HUMAN]",
    Species.MOUSE: "[MOUSE]",
    Species.CAMEL: "[CAMEL]",
    Species.RAT: "[RAT]",
    Species.RABBIT: "[RABBIT]",
    Species.RHESUS: "[RHESUS]",
    Species.UNKNOWN: "[HUMAN]",  # Default to human
}

# Mapping from our ChainType enum to IgLM chain control tokens.
CHAIN_TYPE_MAP = {
    ChainType.HEAVY: "[HEAVY]",
    ChainType.LIGHT: "[LIGHT]",
    ChainType.UNKNOWN: "[HEAVY]",  # Default to heavy
}
```

- [ ] **Step 4: Fix the `.get()` fallbacks**

There are six `.get(..., "heavy")` / `.get(..., "human")` calls, at lines 117-118, 167-168, and 259-260. Every one must use the bracketed form. Each pair becomes:

```python
        iglm_chain = CHAIN_TYPE_MAP.get(chain_type, "[HEAVY]")
        iglm_species = SPECIES_MAP.get(species, "[HUMAN]")
```

- [ ] **Step 5: Run the token-map tests to verify they pass**

```bash
python -m pytest tests/test_iglm.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Fix `_format_for_model` and `_tokenize` return types**

`_format_for_model` declares `list[str]` but appends `str | None`; `_tokenize` declares `dict[str, torch.Tensor]` but returns a dict of lists. Replace both methods:

```python
    def _format_for_model(self, sequences: list[AntibodySequence]) -> list[str]:
        """Format sequences for IgLM (returns raw sequences)."""
        return [seq.primary_chain for seq in sequences]

    def _tokenize(self, formatted_sequences: list[str]) -> dict[str, list[str]]:
        """Tokenization is handled internally by IgLM."""
        return {"sequences": formatted_sequences}
```

Check whether the `dict[str, list[str]]` return type conflicts with the abstract signature in `GenerativeAbLM`:

```bash
grep -n "_tokenize" -A 4 src/ablms/core/generative.py
```

If the base class declares `dict[str, torch.Tensor]`, widen the base to `dict[str, Any]` and note in its docstring that generative models may return non-tensor payloads. Do not silence the mismatch with a cast.

- [ ] **Step 7: Fix the `generate` call and its return handling**

Replace the body of the generation loop in `_generate` (lines 120-150). `generate()` returns `list[str]` and deduplicates internally, so call it once for all sequences instead of N times. It supplies no scores, so score each result with `log_likelihood`.

```python
        # IgLM returns a de-duplicated list of sequences and no scores, so ask
        # for all of them in one call and score them separately.
        generated_seqs = self._iglm.generate(
            chain_token=iglm_chain,
            species_token=iglm_species,
            prompt_sequence=prompt,
            num_to_generate=num_sequences,
            top_p=top_p if top_p is not None else 1.0,
            temperature=temperature,
            **kwargs,
        )

        sequences = []
        scores = []
        for generated_seq in generated_seqs:
            if chain_type == ChainType.LIGHT:
                ab_seq = AntibodySequence(light=generated_seq, species=species)
            else:
                ab_seq = AntibodySequence(heavy=generated_seq, species=species)
            sequences.append(ab_seq)
            scores.append(
                self._iglm.log_likelihood(
                    sequence=generated_seq,
                    chain_token=iglm_chain,
                    species_token=iglm_species,
                )
            )

        return sequences, scores
```

`_generate` still accepts `top_k` and `max_length` in its signature because `GenerativeAbLM` defines them, but `iglm.generate` supports neither. Add this immediately after the two `iglm_*` assignments so callers are not silently misled:

```python
        if top_k is not None:
            raise UnsupportedOperationError(
                "IgLM does not support top_k sampling; use top_p instead."
            )
        if max_length is not None:
            raise UnsupportedOperationError(
                "IgLM does not support a max_length argument."
            )
```

Add `UnsupportedOperationError` to the existing import from `ablms.exceptions`.

- [ ] **Step 8: Fix both `infill` calls**

In the `mask_range is not None` branch, fix the call. (Task 2 already deleted the unused `prefix` and `suffix` locals from this block.) Replace the loop with:

```python
            start, end = mask_range

            infilled_seqs = self._iglm.infill(
                sequence=seq_str,
                chain_token=iglm_chain,
                species_token=iglm_species,
                infill_range=(start, end),
                temperature=temperature,
                num_to_generate=num_sequences,
                **kwargs,
            )

            for infilled_seq in infilled_seqs:
                if chain_type == ChainType.LIGHT:
                    ab_seq = AntibodySequence(light=infilled_seq, species=species)
                else:
                    ab_seq = AntibodySequence(heavy=infilled_seq, species=species)
                sequences.append(ab_seq)
                scores.append(
                    self._iglm.log_likelihood(
                        sequence=infilled_seq,
                        chain_token=iglm_chain,
                        species_token=iglm_species,
                    )
                )
```

Apply the same shape to the second `infill` call in the `elif sequence.is_masked:` branch (lines 216-238), keeping its `sequence=seq_str.replace(mask_token, "")` argument.

- [ ] **Step 9: Use `primary_chain` at the three `seq_str` sites**

Lines 164 and 257 both read `seq_str = sequence.heavy_chain or sequence.light_chain`. Replace each with:

```python
        seq_str = sequence.primary_chain
```

This resolves the remaining `replace`-on-`None` and `not-subscriptable` diagnostics in this file.

- [ ] **Step 10: Fix the `log_likelihood` call in `_compute_log_likelihood`**

```python
        score = self._iglm.log_likelihood(
            sequence=seq_str,
            chain_token=iglm_chain,
            species_token=iglm_species,
        )
```

- [ ] **Step 11: Verify iglm.py is clean for both tools**

```bash
ty check src/ablms/generators/iglm.py
ruff check src/ablms/generators/iglm.py
black --check src/ablms/generators/iglm.py
python -m pytest tests/test_iglm.py -q
```

Expected: `All checks passed!` from ty and ruff, black clean, and all `test_iglm.py` tests passing.

```bash
ty check src/ 2>&1 | tail -1
```

Expected: `Found 28 diagnostics` (41 after Task 4, minus the 13 remaining in `iglm.py`).

- [ ] **Step 12: Commit**

```bash
git add src/ablms/generators/iglm.py tests/test_iglm.py
git commit -m "Fix IgLM wrapper call signatures and token maps"
```

---

### Task 6: Fix the remaining ty diagnostics

28 diagnostics across seven files. Each group below is an independent fix; verify after each.

**Files:**
- Modify: `src/ablms/parallel/executor.py`, `src/ablms/parallel/worker.py`
- Modify: `src/ablms/outputs/mask_scan.py`, `src/ablms/outputs/generation.py`
- Modify: `src/ablms/encoders/ablang.py`, `src/ablms/encoders/ablang2.py`, `src/ablms/encoders/esm2.py`, `src/ablms/encoders/ftesm.py`, `src/ablms/encoders/igt5.py`
- Modify: `src/ablms/core/base.py`, `src/ablms/core/config.py`

**Interfaces:**
- Consumes: Tasks 3-5.
- Produces: `ty check src/` exits 0. Adds `MaskScanOutput._accuracy_values`, `_perplexity_values`, `_entropy_values`, and `_combined_mask`, all private.

- [ ] **Step 1: Fix the `SpawnContext` annotations**

`torch.multiprocessing` does not re-export the `context` submodule, so `mp.context.SpawnContext` is unresolvable. Import the real type instead.

In both `src/ablms/parallel/executor.py` and `src/ablms/parallel/worker.py`, add to the existing `if TYPE_CHECKING:` block:

```python
    from multiprocessing.context import SpawnContext
```

Then in `executor.py:113`:

```python
        self._mp_context: SpawnContext | None = None
```

And in `worker.py:148`:

```python
        ctx: SpawnContext,
```

- [ ] **Step 2: Fix the `WorkerHandle.process` annotation this reveals**

Now that ty knows `ctx` is a real `SpawnContext`, `ctx.Process(...)` is a `SpawnProcess`, and `WorkerHandle.__init__` declaring `mp.Process` becomes a genuine mismatch. The executor only ever uses a spawn context, so `SpawnProcess` is the accurate type. Find the declaration:

```bash
grep -n "process" src/ablms/parallel/worker.py | grep -i "mp.Process\|: Process"
```

Change that annotation to `SpawnProcess`, and add to `worker.py`'s `TYPE_CHECKING` block:

```python
    from multiprocessing.context import SpawnProcess
```

- [ ] **Step 3: Narrow the lazily-initialised executor attributes**

`self._workers` and `self._result_queue` are `| None` until `_ensure_initialized()` runs, and five sites dereference them without narrowing. Add two accessors to `MultiGPUExecutor` that state the precondition once:

```python
    def _require_workers(self) -> list[WorkerHandle]:
        """The worker handles, which exist only after initialization."""
        if self._workers is None:
            raise RuntimeError(
                "Workers have not been initialized; call _ensure_initialized() first."
            )
        return self._workers

    def _require_result_queue(self) -> mp.Queue:
        """The result queue, which exists only after initialization."""
        if self._result_queue is None:
            raise RuntimeError(
                "Result queue has not been initialized; call _ensure_initialized() first."
            )
        return self._result_queue
```

Then replace the five dereference sites. In `execute_iter`, bind once near `total = len(batches)`:

```python
        total = len(batches)
        workers = self._require_workers()
        result_queue = self._require_result_queue()
        num_workers = len(workers)
```

and use `workers[worker_idx].submit_task(...)` at line 334 and `result_queue.get(timeout=WORKER_TIMEOUT)` at line 357. In `_drain_inflight` (line 414) use `self._require_result_queue().get(timeout=WORKER_TIMEOUT)`. In `_stalled_error` (line 442) use:

```python
        dead = [w.worker_id for w in self._require_workers() if not w.is_alive]
```

- [ ] **Step 4: Verify parallel/ is clean and its tests pass**

```bash
ty check src/ablms/parallel/
python -m pytest tests/test_executor.py -q 2>&1 | tail -2
```

Expected: `All checks passed!` and `14 passed`.

- [ ] **Step 5: Extract typed value helpers in mask_scan.py**

`accuracy()`, `perplexity()`, and `entropy()` return `torch.Tensor | float`, so the three `get_chain_*` methods cannot subscript their results. Extract the per-position computation, which is always a tensor, and share the mask-combining logic that is currently duplicated six times.

Add these four private methods to `MaskScanOutput`, before `accuracy()`:

```python
    def _combined_mask(self, mask: torch.Tensor | None) -> torch.Tensor:
        """Combine the attention mask with an optional user mask."""
        if mask is None:
            return self.attention_mask
        return self.attention_mask & mask.to(self.attention_mask.device)

    def _accuracy_values(self, combined_mask: torch.Tensor) -> torch.Tensor:
        """Per-position accuracy, zeroed at invalid positions."""
        correct = (self.predictions == self.original_token_ids).float()
        return correct * combined_mask.float()

    def _perplexity_values(self, combined_mask: torch.Tensor) -> torch.Tensor:
        """Per-position perplexity, zeroed at invalid positions."""
        original_log_probs = self.log_probabilities.gather(
            dim=-1, index=self.original_token_ids.unsqueeze(-1)
        ).squeeze(-1)
        return torch.exp(-original_log_probs) * combined_mask.float()

    def _entropy_values(self, combined_mask: torch.Tensor) -> torch.Tensor:
        """Per-position entropy, zeroed at invalid positions."""
        ent = -torch.sum(self.probabilities * self.log_probabilities, dim=-1)
        return ent * combined_mask.float()
```

- [ ] **Step 6: Rewrite the six public methods to use the helpers**

`accuracy()` keeps its signature and docstring; only its body changes:

```python
        combined_mask = self._combined_mask(mask)
        return self._aggregate(
            self._accuracy_values(combined_mask), combined_mask, agg
        )
```

`perplexity()` and `entropy()` become the same three lines with `_perplexity_values` and `_entropy_values`.

`get_chain_accuracy()` replaces its two slicing lines:

```python
        start, end = self.token_offsets[chain]
        chain_accuracy = self.accuracy()[start:end]
        chain_attn_mask = self.attention_mask[start:end]
```

with:

```python
        start, end = self.token_offsets[chain]
        chain_accuracy = self._accuracy_values(self.attention_mask)[start:end]
        chain_attn_mask = self.attention_mask[start:end]
```

Apply the same substitution in `get_chain_perplexity()` (`_perplexity_values`) and `get_chain_entropy()` (`_entropy_values`).

- [ ] **Step 7: Verify mask_scan is clean and behaviour is unchanged**

```bash
ty check src/ablms/outputs/mask_scan.py
python -m pytest tests/test_mask_scan.py -q 2>&1 | tail -2
```

Expected: `All checks passed!` and `39 passed`. The refactor must be behaviour-preserving; if any test fails, the extraction is wrong, not the test.

- [ ] **Step 8: Fix the `.item()` float-index diagnostics**

`Tensor.item()` is typed as returning `int | float | bool`, so using it as an index fails. Two sites, `src/ablms/encoders/ablang.py:520` and `src/ablms/encoders/ablang2.py:289`, both read:

```python
            original_token = input_ids[i].item()
```

Change both to:

```python
            original_token = int(input_ids[i].item())
```

- [ ] **Step 9: Fix the three `.to(device)` diagnostics**

transformers' `.to()` resolves through a decorator that ty reads as an unbound `_Wrapped.__call__`, so it reports `Expected PreTrainedModel, found device`. Restructuring the assignment does not help; annotating the local as `Any` does. In `src/ablms/encoders/esm2.py:91-96`:

```python
    def _load_model(self) -> None:
        """Load the model and tokenizer from HuggingFace."""
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        # `Any`: transformers wraps `.to()` in a decorator that static analysis
        # reads as an unbound method.
        model: Any = EsmForMaskedLM.from_pretrained(self._model_id)
        model.to(self._primary_device)
        model.eval()
        self._model = model
```

Add `from typing import Any` if the module lacks it. Apply the same pattern at `src/ablms/encoders/ftesm.py:66` and `src/ablms/encoders/igt5.py:69`, using whichever model class each already calls.

- [ ] **Step 10: Fix the ablang.py lazily-assigned attributes and list types**

In `src/ablms/encoders/ablang.py`, the three `__init__` attributes at lines 73-75 are untyped:

```python
        # Lazy-loaded models
        self._heavy_model: Any = None
        self._light_model: Any = None
        self._ablang_module: Any = None
```

Add `from typing import Any` if absent. Then suppress the one authorised unresolved import at line 81 — `ablang` v1 is genuinely not a declared dependency:

```python
            # ty: ignore[unresolved-import]  # ablang v1 is not a declared
            # dependency; see the tracking issue in Task 8.
            import ablang
```

Verify the comment placement actually suppresses the diagnostic (ty requires it on the offending line, so it may need to be a trailing comment on the `import ablang` line itself):

```bash
ty check src/ablms/encoders/ablang.py --output-format concise
```

Finally, the three `[None] * len(sequences)` lists at lines 302-304 are inferred as `list[None]`, breaking `len(results[0])`, `results[i][layer_idx]`, and `torch.cat(results, ...)`:

```python
        results: list[Any] = [None] * len(sequences)
        masks: list[Any] = [None] * len(sequences)
        offsets: list[Any] = [None] * len(sequences)
```

- [ ] **Step 11: Fix the remaining four single-site diagnostics**

`src/ablms/core/config.py:296` passes `embedding_dim=None` for IgLM, but `ModelConfig.embedding_dim` is `int`. A generative model has no embedding dimension, so the field should be optional. Find and widen the declaration:

```bash
grep -n "embedding_dim" src/ablms/core/config.py | head -5
```

Change the dataclass field to `embedding_dim: int | None`, then check that no consumer assumes non-`None`:

```bash
grep -rn "embedding_dim" src/ablms/ tests/ | grep -v "embedding_dim:" | grep -v "embedding_dim="
```

If a consumer does arithmetic on it, guard that site rather than reverting this change.

`src/ablms/core/base.py:194,198` — `isinstance(sequences[0], str)` does not narrow the list's element type. Make the assumption explicit:

```python
            # Element 0 is representative; a mixed list is a caller error.
            if isinstance(sequences[0], str):
                return [
                    AntibodySequence(heavy=s)
                    for s in cast("list[str]", sequences)
                ]

            if isinstance(sequences[0], AntibodySequence):
                return cast("list[AntibodySequence]", sequences)
```

Add `cast` to the existing `from typing import` line.

`src/ablms/outputs/generation.py:76` — `self.scores[i]` inside the sort key, where `scores` is `list[float] | None`. The guard at line 71 raises when it is `None`, but the closure defeats narrowing. Bind it first:

```python
        scores = self.scores
        if scores is None:
            raise ValueError("Scores are not available for ranking")

        # Sort by score descending
        sorted_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
```

Update the two references below it to use `scores[i]` as well.

- [ ] **Step 12: Verify ty is completely clean**

```bash
ty check src/
```

Expected: `All checks passed!` and exit code 0.

```bash
ruff check src/ tests/ && black --check src/ tests/
python -m pytest -m "not slow" -q 2>&1 | tail -2
```

Expected: all clean, and the full non-slow suite passing.

- [ ] **Step 13: Commit**

```bash
git add src/
git commit -m "Fix remaining ty diagnostics"
```

---

### Task 7: Update project metadata and add the smoke tests

**Files:**
- Modify: `pyproject.toml`
- Modify: `CLAUDE.md`
- Create: `tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: a `smoke` pytest marker and `tests/test_smoke.py`, both consumed by the workflow's `smoke` job in Task 8.

- [ ] **Step 1: Replace mypy with ty in the dev extra and pin the tools**

In `pyproject.toml`:

```toml
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    # Tooling is pinned exactly: ty and ruff are pre-1.0 and change diagnostics
    # between releases, so an unpinned bump would break CI on someone else's
    # release schedule rather than on a change here.
    "black==26.5.1",
    "ruff==0.16.2",
    "ty==0.0.40",
]
```

- [ ] **Step 2: Delete the `[tool.mypy]` block**

Remove these five lines entirely:

```toml
[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
```

- [ ] **Step 3: Register the `smoke` marker**

```toml
markers = [
    "slow: marks tests that load real model weights (deselect with '-m \"not slow\"')",
    "smoke: marks tests that instantiate bundled-weight models offline (implies slow)",
]
```

- [ ] **Step 4: Update CLAUDE.md**

In the "Build and Development Commands" block, replace:

```bash
black src/ tests/
ruff check src/ tests/
mypy src/
```

with:

```bash
black src/ tests/
ruff check src/ tests/
ty check src/
```

- [ ] **Step 5: Write the smoke tests**

These are expected to FAIL right now — both packages are broken under transformers 5.x. That is the point: the job is non-blocking until that is fixed, and the test is what will tell you when it is.

Create `tests/test_smoke.py`:

```python
"""Offline smoke tests for backends that bundle their own weights.

`iglm` and `antiberty` ship weights inside the package, so these run with no
network and no GPU. They exist because every test that loads real weights is
marked `slow` and deselected in CI, which is how these two models came to be
broken under transformers 5.x without the suite noticing.

Both are expected to fail until that incompatibility is resolved. The CI job
running them is non-blocking for exactly that reason.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.smoke, pytest.mark.slow]


def test_iglm_generates_one_sequence():
    """IgLM's small checkpoint should produce a sequence from control tokens."""
    iglm = pytest.importorskip("iglm")
    model = iglm.IgLM(model_name="IgLM-S")
    generated = model.generate(
        chain_token="[HEAVY]",
        species_token="[HUMAN]",
        num_to_generate=1,
        temperature=1.0,
    )
    assert len(generated) == 1
    assert generated[0].isalpha()


def test_antiberty_embeds_one_sequence():
    """AntiBERTy should return an embedding for a single heavy chain."""
    antiberty = pytest.importorskip("antiberty")
    runner = antiberty.AntiBERTyRunner()
    embeddings = runner.embed(["EVQLVESGGGLVQPGRSLRLSCAASGFTFSDYAMH"])
    assert len(embeddings) == 1
    assert embeddings[0].shape[-1] == 512
```

- [ ] **Step 6: Confirm the smoke tests are collected but excluded from the normal run**

```bash
python -m pytest -m smoke --collect-only -q 2>&1 | tail -3
python -m pytest -m "not slow" --collect-only -q 2>&1 | tail -2
```

Expected: the first collects 2 tests; the second does **not** include `test_smoke.py`, because `smoke` tests are also marked `slow`.

- [ ] **Step 7: Confirm the smoke tests fail for the documented reason**

```bash
python -m pytest -m smoke -q 2>&1 | tail -15
```

Expected: both fail — IgLM with `AssertionError: Unrecognized token supplied in starting tokens`, AntiBERTy with `AttributeError: 'AntiBERTy' object has no attribute 'all_tied_weights_keys'`. If either *passes*, the transformers incompatibility has been resolved upstream; record that and flag it, since the smoke job could then start blocking immediately.

- [ ] **Step 8: Verify the whole gate is clean**

```bash
ruff check src/ tests/ && black --check src/ tests/ && ty check src/
python -m pytest -m "not slow" -q 2>&1 | tail -2
```

Expected: all four clean.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml CLAUDE.md tests/test_smoke.py
git commit -m "Replace mypy with ty, pin dev tools, add offline smoke tests"
```

---

### Task 8: Add the CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: the `smoke` marker from Task 7 and the clean state from Tasks 2-6.
- Produces: five job runs on every pull request.

- [ ] **Step 1: Write the workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]

# CI only reads the checkout.
permissions:
  contents: read

# Supersede in-flight runs for the same ref.
concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint:
    name: lint (ruff, black)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      # No project dependencies needed, so skip the editable install entirely.
      - run: pip install ruff==0.16.2 black==26.5.1
      - run: ruff check src/ tests/
      - run: black --check src/ tests/

  typecheck:
    name: typecheck (ty)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
          cache-dependency-path: pyproject.toml
      # ty resolves third-party imports through the environment, so the real
      # dependencies must be installed. The CPU index avoids a ~2.5 GB CUDA wheel.
      - run: pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev]"
      - run: ty check src/ --output-format github

  test:
    name: test (py${{ matrix.python-version }})
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
          cache-dependency-path: pyproject.toml
      - run: pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev]"
      - run: python -m pytest -m "not slow"

  smoke:
    name: smoke (bundled weights, non-blocking)
    runs-on: ubuntu-latest
    # IgLM and AntiBERTy are broken under transformers 5.x. This job reports
    # that breakage without blocking merges; make it required once the
    # transformers-compatibility work lands.
    continue-on-error: true
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
          cache-dependency-path: pyproject.toml
      - run: pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev]"
      - run: python -m pytest -m smoke
```

`fail-fast: false` is set on the matrix so that a failure on 3.10 does not cancel the 3.11 and 3.12 legs — you want to see which versions are affected.

- [ ] **Step 2: Validate the YAML parses**

```bash
python -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('.github/workflows/ci.yml').read_text()); print('valid YAML')"
```

Expected: `valid YAML`. If `yaml` is unavailable, `pip install pyyaml` first.

- [ ] **Step 3: Confirm each job's command passes locally**

The workflow itself cannot be run locally, but every command in it can:

```bash
ruff check src/ tests/
black --check src/ tests/
ty check src/
python -m pytest -m "not slow"
```

Expected: all four pass. (`python -m pytest -m smoke` is expected to fail; that job is non-blocking.)

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "Add CI workflow for lint, typecheck, and tests"
```

- [ ] **Step 5: Push and open the pull request**

```bash
git push -u origin feature/github-actions-ci
gh pr create --fill --title "Add automated CI via GitHub Actions"
```

- [ ] **Step 6: Watch the run and confirm the results**

```bash
gh pr checks --watch
```

Expected: `lint`, `typecheck`, and all three `test` legs pass. `smoke` fails but does not block, because of `continue-on-error`.

If a `test` leg fails on 3.10 or 3.11 but passes on 3.12, that is a genuine version-floor problem this CI was built to catch — fix it rather than narrowing the matrix.

- [ ] **Step 7: File the tracking issues**

Two suppressions and one non-blocking job need something to point at:

```bash
gh issue create --title "IgLM and AntiBERTy are non-functional under transformers 5.x" --body "$(cat <<'EOF'
Both bundled-weight backends fail under transformers 5.x:

- **IgLM**: `BertTokenizerFast(vocab_file=...)` no longer loads the vocabulary, leaving the
  tokenizer with 5 tokens. `[HEAVY]`, `[HUMAN]`, and even the amino acid `E` map to `[UNK]`, so
  `generate()` fails its own assertion: `Unrecognized token supplied in starting tokens`.
  Confirmed under transformers 5.15.0 and 5.3.0.
- **AntiBERTy**: `AttributeError: 'AntiBERTy' object has no attribute 'all_tied_weights_keys'`
  under transformers 5.15.0.

Both packages declare unbounded requirements (`iglm` wants `transformers>=4.6.1`, `antiberty`
wants `>=4.5.1`), which is how they drifted into breakage unnoticed.

The likely fix is a `transformers>=4.30.0,<5` upper bound in `pyproject.toml`, but that must be
validated against all ten models before it can land. `tests/test_smoke.py` covers both cases and
the CI `smoke` job is `continue-on-error: true` until this is resolved.

See `docs/superpowers/specs/2026-08-10-github-actions-ci-design.md`.
EOF
)"

gh issue create --title "ablang (v1) is an undeclared dependency" --body "$(cat <<'EOF'
`src/ablms/encoders/ablang.py` imports `ablang` (v1), but it is not listed in
`project.dependencies` and is not installed in the development environment. Consequences:

- ty reports `unresolved-import`, currently suppressed with a single
  `# ty: ignore[unresolved-import]` at the import site.
- The AbLang v1 slow tests cannot run anywhere.

Decide whether to declare the dependency or drop the encoder, then remove the suppression.
EOF
)"
```

Then update the suppression comment in `src/ablms/encoders/ablang.py` to cite the real issue number, and commit:

```bash
git add src/ablms/encoders/ablang.py
git commit -m "Reference tracking issue in ablang ty suppression"
git push
```

- [ ] **Step 8: Make the checks required**

Once the run is green, enable branch protection on `main` requiring `lint`, `typecheck`, and the three `test` legs. Do **not** include `smoke`. This is a repository-settings change, not a code change:

```bash
gh api -X PUT repos/:owner/:repo/branches/main/protection/required_status_checks \
  -f strict=true \
  -f 'contexts[]=lint (ruff, black)' \
  -f 'contexts[]=typecheck (ty)' \
  -f 'contexts[]=test (py3.10)' \
  -f 'contexts[]=test (py3.11)' \
  -f 'contexts[]=test (py3.12)'
```

If this fails for permissions reasons, configure it through the repository settings UI instead and note that it still needs doing.

---

## Self-review

**Spec coverage.** Every section of the spec maps to a task: workflow structure and installation → Task 8; test invocation → Task 8 (`-m "not slow"` on the command line, `addopts` untouched); dependency versions and tool pinning → Task 7; type-checker consolidation → Task 7; ty scope (`src/` only) → the global constraints and Task 8's command; the 46-diagnostic distribution → Tasks 3-6; the authorised `ablang` suppression → Task 6 Step 10 with the issue in Task 8 Step 7; the IgLM bug and its contract test → Task 5; the smoke job → Tasks 7 and 8; both known issues → Task 8 Step 7. The spec's implementation order maps one-to-one onto Tasks 1-8, with its step 0 becoming Task 1.

**Placeholder scan.** No `TBD`, `TODO`, or "add error handling" steps. Three steps direct the implementer to inspect before editing rather than giving a literal patch — Task 2 Step 4 (confirm the six `F841` sites are dead code, not symptoms), Task 6 Step 2 (locate the `WorkerHandle.process` annotation), and Task 6 Step 11 (check `embedding_dim` consumers). These are deliberate: each is a place where a blind edit could hide a real bug, and each names the exact command to run and the exact decision to make.

**Type consistency.** `AntibodySequence.primary_chain -> str` is defined in Task 4 Step 3 and consumed in Task 4 Step 5 and Task 5 Steps 6 and 9 under that exact name. `SPECIES_MAP` and `CHAIN_TYPE_MAP` keep their names throughout, changing only their values. The `MaskScanOutput` helpers introduced in Task 6 Step 5 (`_combined_mask`, `_accuracy_values`, `_perplexity_values`, `_entropy_values`) are used under those exact names in Step 6. `_require_workers` and `_require_result_queue` from Task 6 Step 3 are used under those names in the same step. `SpawnContext` and `SpawnProcess` are imported in Task 6 Steps 1 and 2 before use.

**Diagnostic arithmetic.** 204 → 44 (Task 3) → 41 (Task 4, three `replace` sites) → 28 (Task 5, thirteen in `iglm.py`) → 0 (Task 6). The plan originally predicted 46/43 at the first two steps; Task 2's dead-code deletion removed two `iglm.py` diagnostics for free, shifting both by two without changing Task 5's end state. The 28 in Task 6 break down as: `executor.py` 5 + `worker.py` 1 + the `SpawnProcess` follow-on, `mask_scan.py` 6, `ablang.py` 8, `ablang2.py` 1, `esm2.py` 1, `ftesm.py` 1, `igt5.py` 1, `base.py` 2, `config.py` 1, `generation.py` 1. Task 6 Step 12 is the checkpoint that catches any drift.
