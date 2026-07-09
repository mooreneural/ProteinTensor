# Contributing to ProteinTensor

Thanks for your interest in ProteinTensor - an AI-native tensor format (`.ptt`)
for protein-structure machine learning. Contributions of all kinds are welcome:
bug reports, documentation, new model adapters, new cached feature types,
benchmarks, and performance work.

## Ways to contribute

- **Report a bug** or request a feature via the issue templates.
- **Improve docs** - the README, docstrings, or examples.
- **Add a model adapter** - convert a `.ptt` into another model's native input
  (see `proteintensor/adapters/`).
- **Add a cached feature** or storage improvement (see `proteintensor/pairs.py`,
  `writer.py`, `reader.py`).
- **Add or improve a benchmark** under `benchmarks/`.

## Development setup

```bash
git clone https://github.com/mooreneural/ProteinTensor
cd ProteinTensor
pip install -e ".[dev]"          # core + dev tools
pip install -e ".[dev,cloud]"    # add S3/GCS streaming deps
pip install -e ".[ligands]"      # add RDKit for SMILES ligand support
```

Requires Python >= 3.9.

## Running the tests

```bash
pytest tests/ -v
```

All tests must pass before a pull request can be merged. If you add or change
behavior, add a test for it. Tests should be self-contained and must not require
a GPU, network access, or a real cloud account (cloud paths are tested with the
in-memory `memory://` fsspec backend).

## Benchmarks

Benchmarks live in `benchmarks/` and write results to `benchmarks/results/` plus
a regenerated markdown summary. If your change affects performance, run the
relevant benchmark and include the numbers in your PR.

**Benchmark honesty is a project value.** A benchmark must *measure*, not
estimate. Compare against a fair, optimized baseline, state the hardware, and
label any projection as a projection - never as a measurement. If a change makes
a previously reported number look worse because the baseline got fairer, that is
a correct result and should be reported as such.

## Code style

- Match the style of the surrounding code: type hints, clear docstrings, and
  descriptive names.
- Use a hyphen (`-`), not an em dash, in code, comments, docs, and CLI text.
- Keep public API changes documented in the README and exported from
  `proteintensor/__init__.py`.
- Keep the `.ptt` format backward compatible where possible; bump
  `FORMAT_VERSION` in `schema.py` when the on-disk layout changes.

## Pull request process

1. Fork the repo and create a branch from `main`.
2. Make your change with tests and, if relevant, benchmark numbers.
3. Ensure `pytest tests/ -v` passes.
4. Open a pull request using the template, describing what changed and why, and
   noting any measured results.

## Reporting security issues

Please do not open public issues for security vulnerabilities. See
[SECURITY.md](SECURITY.md) for how to report them privately.

## Code of Conduct

By participating, you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).
