# Release: synth-containers

Release this package from the root of this standalone repository. Tested
pairing: `synth-optimizers==0.2.22` pins `synth-containers==0.4.3`.

## Build

Run from the repository root:

```bash
uv run --group dev pytest tests
uv run --group dev ruff check src
python3 scripts/check-type-debt.py
uv build
uv run --group dev twine check dist/*
```

The eight historical metadata/reward failures have been reconciled with the
current contracts. Missing rewards are still asserted before completion;
terminal and recovered catalogs now test both scored and omitted rewards.
The publish workflow runs the entire suite without deselections. Full-suite
verification remains mandatory, including timing-sensitive annotation tests.
The type-debt gate retains an explicit 173-diagnostic historical baseline;
the integrated candidate currently reports 172 diagnostics,
mostly `invalid-argument-type` and `unresolved-attribute`; annotating that
surface is separate work. Neither is a licence to add more.

## Cookbook compile check

Runnable cookbooks live in the separate
[`synth-cookbooks-public`](https://github.com/synth-laboratories/synth-cookbooks-public)
repository. For cookbook-facing releases, clone it next to this checkout and
compile the touched cookbook entrypoints against this source tree:

```bash
git clone https://github.com/synth-laboratories/synth-cookbooks-public.git ../synth-cookbooks-public
PYTHONPATH=src python -m py_compile $(rg --files ../synth-cookbooks-public/cookbooks -g '*.py')
```

## Changelog

- Update `changelog.log` in the same change that updates package version or release docs.
- Organize entries by day: `## YYYY-MM-DD`.
- Keep the file terse: about 20 total lines for the current daily-dev window.
- Use bullets only; no paragraphs, migration snippets, or install code blocks.
- Prefer shipped user-facing changes over implementation narration.
- Link merged PRs where available, for example `[PR #2](https://github.com/synth-laboratories/containers/pull/2)`.
- Include the PyPI version in one bullet when a package was published.
- Keep unreleased or blocked work explicit and short.

## Publish

After confirming the version and inspecting the generated artifacts:

```bash
uv publish dist/*
```

This repository no longer carries a GitHub Actions publish workflow; publish
only artifacts built from a tree that passed every gate above. Never create a
public version tag before all release gates pass. Published versions are
immutable: `0.4.3` (tag `v0.4.3`) is the current published release and the
version `synth-optimizers==0.2.22` pins; `0.4.2` remains published unchanged.
