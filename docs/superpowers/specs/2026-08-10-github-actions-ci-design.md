# Automated CI via GitHub Actions

**Date:** 2026-08-10
**Status:** Awaiting review

## Problem

The repository has no CI. The only workflow is `.github/workflows/python-publish.yaml`, which
fires on release creation and does nothing but build and publish to PyPI. Nothing runs on a pull
request, so a change can merge with failing tests, unsorted imports, or a type error.

Adding a gate is complicated by the fact that **nothing in the repository is currently clean**:

| Tool | Current state |
| --- | --- |
| `ty check src/` | 204 diagnostics |
| `mypy src/` | 233 errors |
| `ruff check src/ tests/` | 57 errors (41 auto-fixable) |
| `black --check src/ tests/` | 18 files would reformat |

A blocking gate therefore cannot simply be switched on; the cleanup is part of the work.

### The relevant facts about the existing suite

**The `slow` marker is already the exact selector CI needs.** 139 of 408 tests carry
`@pytest.mark.slow`, defined in `pyproject.toml` as "marks tests that load real model weights".
There are no CUDA references anywhere in `tests/` — no test requests a GPU, so `slow` is the
only axis that needs deselecting. `pytest -m "not slow"` yields 253 passed / 16 skipped.

**Test execution is not the cost.** That suite runs in **1.8 seconds**. CI wall time will be
almost entirely dependency installation, which is where the optimisation effort belongs.

**The heavy model backends are imported lazily.** `antiberty`, `ablang2`, `ablang`, and `iglm`
are imported inside `_load_model()` bodies (`encoders/antiberty.py:59`, `encoders/ablang2.py:65`,
`encoders/ablang.py:81`, `generators/iglm.py:74`), not at module scope, which is why the non-slow
suite passes without ever touching model weights.

**Almost all ty noise has a single root cause.** `core/base.py:83-84` assigns
`self._model = None` and `self._tokenizer = None` with no annotation, so ty infers
`Unknown | None` and flags every downstream attribute access and call across every encoder.
Annotating both as `Any` takes ty from **204 to 46** diagnostics. This is honest typing rather
than a workaround: HuggingFace model objects are dynamically shaped, and the code legitimately
reaches for `self._model.bert` and `self._tokenizer.sep_token_id`, which no static
`PreTrainedModel` type declares.

## Validation performed before writing this spec

Two assumptions were checked empirically rather than trusted, because both could have invalidated
the design:

1. **ty's diagnostic count is stable across dependency versions.** `ty check src/` reports
   exactly 204 diagnostics in the local environment (torch 2.11, transformers 5.3) *and* in a
   clean virtualenv resolved to latest (torch 2.13.0+cpu, transformers 5.15.0). This is what
   makes a hard type gate compatible with unpinned dependencies; had the count drifted, the gate
   would have needed a constraints file.
2. **The dependency set installs cleanly from wheels.** A clean-venv resolution of
   `ablms[dev]` with `--extra-index-url https://download.pytorch.org/whl/cpu` produces ~60
   packages, all wheels, no sdist builds, with `torch-2.13.0+cpu`. The non-slow suite passes in
   that environment (253 passed, 16 skipped, 1.8s).

## Goals

- Every pull request and every push to `main` runs formatting, lint, type checking, and the
  non-slow test suite.
- All four checks pass from the first green run, so they can be made required branch checks
  immediately rather than being advisory checks that decay.
- CI installs CPU-only torch, never the CUDA build.
- A bare `pytest` locally still runs the full suite, including slow tests.
- One type checker in the project, not two that disagree.

## Non-goals

- No GPU or slow-test job. There is no self-hosted runner, and the slow tests download
  multi-gigabyte weights.
- No coverage reporting. `pytest-cov` is in the dev extra but no threshold or upload service is
  configured; adding one is a separate decision.
- No HuggingFace weight caching. The non-slow suite does not need weights.
- Not fixing the undeclared `ablang` (v1) dependency. See "Known issues left open".

## Design

### Workflow structure

A single new file, `.github/workflows/ci.yml`, triggered on `pull_request` and on `push` to
`main`, with a concurrency group keyed on the ref so superseded runs cancel and a top-level
`permissions: contents: read` — CI only needs to read the checkout, and the existing publish
workflow already shows the pattern of declaring permissions explicitly. Three jobs run in
parallel:

| Job | Python | Installs | Command |
| --- | --- | --- | --- |
| `lint` | 3.12 | `ruff`, `black` only | `ruff check src/ tests/`, `black --check src/ tests/` |
| `typecheck` | 3.12 | project + `ty` | `ty check src/ --output-format github` |
| `test` | 3.10, 3.11, 3.12 | project | `pytest -m "not slow"` |

`lint` installs no project dependencies and finishes in seconds. `typecheck` must install the
real dependencies, because ty resolves third-party imports through the environment — without
`torch` and `transformers` present it would report unresolved-import errors instead of useful
diagnostics. It therefore costs the same install as a `test` job, but runs concurrently with
them, so total wall clock is roughly one install.

The Python matrix covers 3.10, 3.11, and 3.12, matching the classifiers in `pyproject.toml`
exactly and exercising the `requires-python = ">=3.10"` floor.

### Installation

The `typecheck` and `test` jobs share this pattern (`typecheck` hardcodes `"3.12"` where the
matrix job interpolates `${{ matrix.python-version }}`):

```yaml
- uses: actions/setup-python@v5
  with:
    python-version: ${{ matrix.python-version }}
    cache: pip
    cache-dependency-path: pyproject.toml
- run: pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev]"
```

`--extra-index-url` pointed at the PyTorch CPU index is what avoids the ~2.5 GB CUDA wheel.
Verified to resolve `torch-2.13.0+cpu`.

`lint` does not use this pattern at all: it needs no project dependencies, so it installs the two
formatters directly (`pip install ruff==... black==...`) and skips the editable install entirely.

### Test invocation

`pyproject.toml`'s `addopts = "-v --tb=short"` stays as it is. CI passes `-m "not slow"` on the
command line rather than adding it to `addopts`, so that running `pytest` locally continues to
execute the full suite by default. Deselecting slow tests is a CI policy, not a project default.

### Dependency versions

Dependencies stay unpinned, so CI doubles as an early-warning system for upstream breakage — the
right trade for a library that wraps `transformers`. The validation above establishes that this
does not destabilise the type gate.

**Development tooling is the exception and gets exact pins.** The rule is: runtime dependencies
float, because their breakage is news we want; the tools doing the checking are pinned, because
their breakage is noise. Concretely, the dev extra becomes `ty==0.0.40`, `ruff==0.16.2`, and
`black==26.5.1` instead of the current `ruff>=0.1.0` and `black>=23.0.0`. Both ty and ruff are
pre-1.0 and change diagnostics between releases; unpinned, either would turn a required check red
on Astral's release schedule rather than on a change to this repository. Black is pinned for the
same reason at one remove — its stable style can shift in a new year's release, which would
reformat files and fail `--check` with no local change.

Bumping these is then a deliberate, reviewable commit rather than a surprise.

### Type checker consolidation

`mypy` is removed from the project: the `[tool.mypy]` block comes out of `pyproject.toml` and
`mypy>=1.0.0` comes out of the dev extra. It reports 233 errors, is fully redundant with ty, and
keeping both means maintaining two disagreeing sets of suppressions. `ty` replaces it in the dev
extra. Note that `CLAUDE.md` documents `mypy src/` as a development command and must be updated
in the same change.

### ty scope

The gate covers `src/` only. `ty check src/ tests/` reports 119 diagnostics after the `base.py`
annotation — 73 of them in test code, concentrated in `test_mask_scan.py` (31) and
`test_executor.py` (25). Type errors in test code are lower-value than in library code, and
widening the gate to `tests/` is a reasonable follow-up once it is established.

### All 46 remaining diagnostics are first-party

Every one of the 46 diagnostics left after the `base.py` annotation has its primary span in
`src/`; none are inside `site-packages` or typeshed. (ty's default output prints `-->` markers for
secondary context locations as well as the primary one, which makes dependency files *look* like
error sites when counting by hand. `--output-format concise` gives one line per diagnostic and is
the count to trust.) The practical consequence is that no rule suppression or `[tool.ty.rules]`
configuration is needed — the gate can be reached by fixing real code.

The distribution:

| File | Count |
| --- | --- |
| `generators/iglm.py` | 15 |
| `encoders/ablang.py` | 8 |
| `parallel/executor.py` | 6 |
| `outputs/mask_scan.py` | 6 |
| `encoders/esm2.py`, `core/base.py` | 2 each |
| `encoders/{ablang2,antiberty,ftesm,igt5}.py`, `core/config.py`, `outputs/generation.py`, `parallel/worker.py` | 1 each |

**One diagnostic is a deliberate exception.** `encoders/ablang.py:81` reports
`unresolved-import: Cannot resolve imported module 'ablang'` because AbLang v1 is genuinely not a
declared dependency (see "Known issues left open"). Since resolving that question is out of scope,
this single line gets a narrow `# ty: ignore[unresolved-import]` with a comment pointing at the
issue — the one suppression in the codebase, scoped to one rule on one line, rather than a global
rule change.

## The IgLM bug

Running ty surfaced a genuine defect that the type gate pays for by itself.
`src/ablms/generators/iglm.py` calls the `iglm` package with keyword names that do not exist:

| Call site | Wrapper passes | Actual signature |
| --- | --- | --- |
| `iglm.py:125-128` | `generate(chain_type=, species=, prompt=)` | `generate(chain_token, species_token, prompt_sequence=None, num_to_generate=1000, top_p=1, temperature=1)` |
| `iglm.py:181-185` | `infill(sequence=, chain_type=, species=, infill_range=)` | `infill(sequence, chain_token, species_token, infill_range, ...)` |
| `iglm.py:263-266` | `log_likelihood(sequence=, chain_type=, species=)` | `log_likelihood(sequence, chain_token, species_token, infill_range=None)` |

The map *values* are wrong as well. `SPECIES_MAP` (`iglm.py:14-22`) and `CHAIN_TYPE_MAP`
(`iglm.py:25-29`) produce `"human"` and `"heavy"`, but IgLM passes these to
`tokenizer.convert_tokens_to_ids()` and asserts the result is not the unknown token. They must be
vocabulary control tokens. IgLM's `vocab.txt` defines exactly: `[CAMEL]`, `[HUMAN]`, `[MOUSE]`,
`[RABBIT]`, `[RAT]`, `[RHESUS]`, `[HEAVY]`, `[LIGHT]`.

**Every IgLM generate, infill, and scoring call raises `TypeError` today.** There is no
`tests/test_iglm.py` at all, which is why this went unnoticed.

Both bugs are fixed as part of this work — 15 of the 46 remaining diagnostics are in this one
file, so a green gate on `src/` requires it regardless.

### Guarding against recurrence

A new `tests/test_iglm.py` adds a signature-contract test in the established style of
`tests/test_encoder_contract.py`, which already uses `inspect.signature` to enforce cross-class
invariants. It asserts two things:

1. The keyword arguments the wrapper passes bind successfully against
   `inspect.signature(IgLM.generate)`, `IgLM.infill`, and `IgLM.log_likelihood`.
2. Every value in `SPECIES_MAP` and `CHAIN_TYPE_MAP` is a real token in IgLM's vocabulary.

Both checks run without model weights, network access, or a GPU, so they belong in the non-slow
suite and will run on every PR. This catches the same class of upstream API drift that produced
the bug.

## Implementation order

The cleanup must land before the workflow, or the first CI run is red. Sequenced so each step is
independently verifiable:

0. **Align local tool versions with the pins first.** The local environment has ruff 0.15.6 while
   the pin will be 0.16.2; formatting with a different version than CI enforces is how a
   `--check` gate fails on a clean checkout. Install `ty==0.0.40`, `ruff==0.16.2`,
   `black==26.5.1` before touching any code. (Verified: both versions report the same 57 ruff
   errors and 18 black reformats, so the counts below hold either way.)
1. **Format and lint.** `black src/ tests/` (18 files), `ruff check --fix src/ tests/` (41 of 57
   errors), then the remaining 16 ruff errors by hand.
2. **Annotate the model attributes.** `core/base.py:83-84` to `Any`. Verify ty drops 204 → 46.
3. **Fix the IgLM bug.** Correct keyword names and both token maps; add `tests/test_iglm.py`.
4. **Fix the remaining 31 ty diagnostics**, i.e. all those outside `iglm.py`, distributed as in
   the table above. Several are real latent defects rather than annotation noise:
   `executor.py` dereferences `self._workers` and `self._result_queue` without narrowing their
   `| None` types (lines 325, 334, 357, 414, 442), and `encoders/ablang.py:530` and
   `encoders/ablang2.py:297` subscript a `Tensor` with a `float`. This step also applies the
   single `# ty: ignore[unresolved-import]` for `ablang.py:81`.
5. **Update project metadata.** Remove the `[tool.mypy]` block and `mypy>=1.0.0` from
   `pyproject.toml`; replace `ruff>=0.1.0` and `black>=23.0.0` with exact pins and add
   `ty==0.0.40` to the dev extra; update `CLAUDE.md`, which currently documents `mypy src/` as a
   development command.
6. **Add `.github/workflows/ci.yml`.**

## Testing strategy

Each cleanup step is verified locally before the next begins: `ruff check`, `black --check`,
`ty check src/`, and `pytest -m "not slow"` must all be clean by the end of step 5. The IgLM fix
is verified by its own contract test, which does not require weights.

The workflow itself is verified by opening a pull request and confirming that all five job runs
pass — `lint`, `typecheck`, and the three `test` matrix legs — before the checks are made required
in branch protection. The workflow file cannot be validated locally; a real PR run is the test.

## Known issues left open

- **`ablang` (v1) is an undeclared dependency.** `encoders/ablang.py:81` imports it, but it is
  not in `project.dependencies` and is not installed in the development environment, so ty
  reports `unresolved-import` and the AbLang v1 slow tests cannot run anywhere. Deciding whether
  to declare it or drop the encoder is out of scope; it needs its own issue. This work only
  suppresses that one diagnostic so the gate can be reached, and the suppression comment should
  reference the issue so it is not mistaken for permanent.
- **`tests/test_model_metadata.py` reaches the HuggingFace Hub.** It skips gracefully when the
  Hub is unreachable (`test_model_metadata.py:144-145`), so the worst CI outcome is a skipped
  test rather than a failure. Acceptable as-is, but it means those 16 skips may become passes or
  stay skips depending on Hub reachability and rate limiting from GitHub runners.
